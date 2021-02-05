from collections import namedtuple, defaultdict, deque
import queue

import numpy as np
import torch


INT32_MAX = 2147483647

np.seterr(all='raise')


class LevelSampler():
    def __init__(
        self, 
        seeds, 
        obs_space, 
        action_space, 
        num_actors=1, 
        strategy='random', 
        max_score_coef=0.9,
        replay_schedule='fixed', 
        score_transform='power',
        temperature=1.0, 
        eps=0.05,
        rho=1.0, 
        replay_prob=0.95, 
        alpha=1.0, 
        staleness_coef=0, 
        staleness_transform='power', 
        staleness_temperature=1.0, 
        sample_full_distribution=False, 
        seed_buffer_size=0, 
        seed_buffer_priority='score',
        tscl_window_size=0):
        """
        Inputs: 
            seeds: List, Seeds that can be sampled.
            rho: float, Minimum probability of sampling a replay level. 
                Note math.round(rho * len(seeds)) will first be sampled before sampling replay levels.
            alpha: Smoothing factor for updating scores using an exponential weighted average.
            obs_space: Gym env observation space.
            action_space: Gym env action space.
            strategy: Sampling strategy (random, sequential, policy entropy).
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_actors = num_actors
        self.strategy = strategy
        self.max_score_coef = max_score_coef
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.replay_prob = replay_prob # Replay probability
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature

        # Track seeds and scores as in np arrays backed by shared memory
        self.seed_buffer_size = seed_buffer_size if not seeds else len(seeds)
        N = self.seed_buffer_size
        self._init_seed_index(seeds)

        self.unseen_seed_weights = np.array([1.]*N)
        self.seed_scores = np.array([0.]*N, dtype=np.float)
        self.partial_seed_scores = np.zeros((num_actors, N), dtype=np.float)
        self.partial_seed_max_scores = np.ones((num_actors, N), dtype=np.float)*float('-inf')
        self.partial_seed_steps = np.zeros((num_actors, N), dtype=np.int32)
        self.seed_staleness = np.array([0.]*N, dtype=np.float)

        self.running_sample_count = 0

        self.next_seed_index = 0 # Only used for sequential strategy

        # Only used for infinite seed setting
        self.sample_full_distribution = sample_full_distribution
        if self.sample_full_distribution:
            self.seed2actor = defaultdict(set)
            self.working_seed_buffer_size = 0
            self.seed_buffer_priority = seed_buffer_priority
            self.staging_seed_set = set()
            self.working_seed_set = set()

            self.seed2timestamp_buffer = {} # Buffer seeds are unique across actors
            self.partial_seed_scores_buffer = [{} for _ in range(num_actors)]
            self.partial_seed_max_scores_buffer = [{} for _ in range(num_actors)]
            self.partial_seed_steps_buffer = [{} for _ in range(num_actors)]

        # TSCL specific data structures
        if self.strategy.startswith('tscl'):
            self.tscl_window_size = tscl_window_size
            self.tscl_return_window = [deque(maxlen=self.tscl_window_size) for _ in range(N)]
            self.tscl_episode_window = [deque(maxlen=self.tscl_window_size) for _ in range(N)]
            self.unseen_seed_weights = np.zeros(N) # Force uniform distribution over seeds

    def seed_range(self):
        if not self.sample_full_distribution:
            return (int(min(self.seeds)), int(max(self.seeds)))
        else:
            return (0, INT32_MAX)

    def _init_seed_index(self, seeds):
        if seeds:
            self.seeds = np.array(seeds, dtype=np.int64)
            self.seed2index = {seed: i for i, seed in enumerate(seeds)}
        else:
            self.seeds = np.zeros(self.seed_buffer_size, dtype=np.int64) - 1
            self.seed2index = {}

    @property
    def _proportion_filled(self):
        if self.sample_full_distribution:
            return self.working_seed_buffer_size/self.seed_buffer_size
        else:
            return 0

    def update_with_rollouts(self, rollouts):
        if self.strategy == 'random':
            return

        # Update with a RolloutStorage object
        if self.strategy == 'policy_entropy':
            score_function = self._average_entropy
        elif self.strategy == 'least_confidence':
            score_function = self._average_least_confidence
        elif self.strategy == 'min_margin':
            score_function = self._average_min_margin
        elif self.strategy == 'gae':
            score_function = self._average_gae
        elif self.strategy == 'value_l1':
            score_function = self._average_value_l1
        elif self.strategy == 'one_step_td_error':
            score_function = self._one_step_td_error
        elif self.strategy == 'tscl_window':
            score_function = self._tscl_window
        else:
            raise ValueError(f'Unsupported strategy, {self.strategy}')

        self._update_with_rollouts(rollouts, score_function)

    def update_seed_score(self, actor_index, seed, score, max_score, num_steps):
        if self.sample_full_distribution and seed in self.staging_seed_set:
            score, seed_idx = self._partial_update_seed_score_buffer(actor_index, seed, score, num_steps, done=True)
        else:
            score, seed_idx = self._partial_update_seed_score(actor_index, seed, score, max_score, num_steps, done=True)

    def _partial_update_seed_score(self, actor_index, seed, score, max_score, num_steps, done=False):
        seed_idx = self.seed2index.get(seed, -1)
        if seed_idx < 0: return 0, -1
        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_max_score = self.partial_seed_max_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score)*num_steps/float(running_num_steps)
        merged_max_score = max(partial_max_score, max_score)

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0. # Zero partial score, partial num_steps
            self.partial_seed_max_scores[actor_index][seed_idx] = float('-inf')
            self.partial_seed_steps[actor_index][seed_idx] = 0
            self.unseen_seed_weights[seed_idx] = 0. # No longer unseen
            old_score = self.seed_scores[seed_idx]
            total_score = self.max_score_coef*merged_max_score + (1 - self.max_score_coef)*merged_score
            self.seed_scores[seed_idx] = (1 - self.alpha)*old_score + self.alpha*total_score
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_max_scores[actor_index][seed_idx] = merged_max_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps

        return merged_score, seed_idx

    @property
    def _next_buffer_index(self):
        if self._proportion_filled < 1.0:
            return self.working_seed_buffer_size
        else:
            if self.seed_buffer_priority == 'replay_support':
                return self.sample_weights().argmin()
            else:
                return self.seed_scores.argmin()

    def _partial_update_seed_score_buffer(self, actor_index, seed, score, num_steps, done=False):
        seed_idx = -1
        self.seed2actor[seed].add(actor_index)
        partial_score = self.partial_seed_scores_buffer[actor_index].get(seed, 0)
        partial_num_steps = self.partial_seed_steps_buffer[actor_index].get(seed, 0)

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score)*num_steps/float(running_num_steps)

        if done:
            # Move seed into working seed data structures
            seed_idx = self._next_buffer_index
            if self.seed_scores[seed_idx] <= abs(merged_score):
                self.unseen_seed_weights[seed_idx] = 0. # Unmask this index
                self.working_seed_set.discard(self.seeds[seed_idx])
                self.working_seed_set.add(seed)
                self.seeds[seed_idx] = seed
                self.seed2index[seed] = seed_idx 
                self.seed_scores[seed_idx] = merged_score
                self.partial_seed_scores[:,seed_idx] = 0.
                self.partial_seed_steps[:,seed_idx] = 0 
                self.seed_staleness[seed_idx] = self.running_sample_count - self.seed2timestamp_buffer[seed]
                self.working_seed_buffer_size = min(self.working_seed_buffer_size + 1, self.seed_buffer_size)

            # Zero partial score, partial num_steps, remove seed from staging data structures
            for a in self.seed2actor[seed]:
                self.partial_seed_scores_buffer[a].pop(seed, None) 
                self.partial_seed_steps_buffer[a].pop(seed, None)
            del self.seed2timestamp_buffer[seed]
            del self.seed2actor[seed]
            self.staging_seed_set.remove(seed)
        else:
            self.partial_seed_scores_buffer[actor_index][seed] = merged_score
            self.partial_seed_steps_buffer[actor_index][seed] = running_num_steps

        return merged_score, seed_idx

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        num_actions = self.action_space.n
        max_entropy = -(1./num_actions)*np.log(1./num_actions)*num_actions

        scores = -torch.exp(episode_logits)*episode_logits.sum(-1)/max_entropy
        mean_score = scores.mean().item()
        max_score = scores.max().item()

        return mean_score, max_score

    def _average_least_confidence(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        scores = 1 - torch.exp(episode_logits.max(-1, keepdim=True)[0])

        mean_score = scores.mean().item()
        max_score = scores.max().item()
        
        return mean_score, max_score

    def _average_min_margin(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        scores = top2_confidence[:,0] - top2_confidence[:,1]
        mean_score = 1 - scores.mean().item()
        max_score = 1 - scores.min().item()

        return mean_score, max_score

    def _average_gae(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        mean_score = advantages.mean().item()
        max_score = advantages.max().item()

        return mean_score, max_score

    def _average_value_l1(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        abs_advantages = (returns - value_preds).abs()

        mean_score = abs_advantages.mean().item()
        max_score = abs_advantages.max().item()

        return mean_score, max_score

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs['rewards']
        value_preds = kwargs['value_preds']

        max_t = len(rewards)
        td_errors = (rewards[:-1] + value_preds[:max_t-1] - value_preds[1:max_t]).abs()

        mean_score = td_errors.mean().item()
        max_score = abs_advantages.max().item()

        return mean_score, max_score

    def _tscl_window(self, **kwargs):
        rewards = kwargs['rewards']
        seed = kwargs['seed']

        seed_idx = self.seed2index.get(seed, -1)
        assert(seed_idx >= 0)

        # Add rewards to the seed window
        episode_total_reward = rewards.sum().item()
        self.tscl_return_window[seed_idx].append(episode_total_reward)
        self.tscl_episode_window[seed_idx].append(self.running_sample_count)

        # Compute linear regression coeficient in the window
        x = self.tscl_episode_window[seed]
        y = self.tscl_return_window[seed]
        A = np.vstack([x, np.ones(len(x))]).T
        c,_ = np.linalg.lstsq(A, y, rcond=None)[0]

        c = abs(c)
        return c, c

    @property
    def requires_value_buffers(self):
        return self.strategy in ['gae', 'value_l1', 'one_step_td_error', 'tscl_window']

    @property
    def _has_working_seed_buffer(self):
        return not self.sample_full_distribution or (self.sample_full_distribution and self.seed_buffer_size > 0)

    def _update_with_rollouts(self, rollouts, score_function):
        if not self._has_working_seed_buffer:
            return

        level_seeds = rollouts.level_seeds
        policy_logits = rollouts.action_log_dist
        done = ~(rollouts.masks > 0)
        total_steps, num_actors = policy_logits.shape[:2]
        num_decisions = len(policy_logits)

        for actor_index in range(num_actors):
            done_steps = done[:,actor_index].nonzero()[:total_steps,0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps: break

                if t == 0: # If t is 0, then this done step caused a full update of previous seed last cycle
                    continue 

                seed_t = level_seeds[start_t,actor_index].item()

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:t,actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                score_function_kwargs['seed'] = seed_t

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:t,actor_index]
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:t,actor_index]
                    score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:t,actor_index]

                score, max_score = score_function(**score_function_kwargs)
                num_steps = len(episode_logits)
                self.update_seed_score(actor_index, seed_t, score, max_score, num_steps)

                start_t = t.item()

            if start_t < total_steps:
                seed_t = level_seeds[start_t,actor_index].item()

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:,actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                score_function_kwargs['seed'] = seed_t

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:,actor_index]
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:,actor_index]
                    score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:,actor_index]

                score, max_score = score_function(**score_function_kwargs)
                num_steps = len(episode_logits)

                if self.sample_full_distribution and seed_t in self.staging_seed_set:
                    self._partial_update_seed_score_buffer(actor_index, seed_t, score, num_steps)
                else:
                    self._partial_update_seed_score(actor_index, seed_t, score, max_score, num_steps)

    def after_update(self):
        if not self._has_working_seed_buffer:
            return

        # Reset partial updates, since weights have changed, and thus logits are now stale
        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self.update_seed_score(actor_index, self.seeds[seed_idx], 0, float('-inf'), 0)

        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

        # Likewise, reset partial update buffers
        if self.sample_full_distribution:
            for actor_index in range(self.num_actors):
                actor_staging_seeds = list(self.partial_seed_scores_buffer[actor_index].keys())
                for seed in actor_staging_seeds:
                    if self.partial_seed_scores_buffer[actor_index][seed] > 0:
                        self.update_seed_score(actor_index, seed, 0, float('-inf'), 0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0

    def _sample_replay_level(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float)/len(sample_weights)

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def _sample_unseen_level(self):
        if self.sample_full_distribution:
            seed = int(np.random.randint(1,INT32_MAX))
            # Ensure unique new seed outside of working and staging set
            while seed in self.staging_seed_set or seed in self.working_seed_set:
                seed = int(np.random.randint(1,INT32_MAX))
            self.seed2timestamp_buffer[seed] = self.running_sample_count
            self.staging_seed_set.add(seed)
        else:
            sample_weights = self.unseen_seed_weights/self.unseen_seed_weights.sum()
            seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
            seed = self.seeds[seed_idx]

            self._update_staleness(seed_idx)

        return int(seed)

    def sample(self, strategy=None):
        if strategy == 'full_distribution':
            raise ValueError('One-off sampling via full_distribution strategy is not supported.')

        self.running_sample_count += 1

        if not strategy:
            strategy = self.strategy

        # Sample uniformly from the full distribution (seeds from 0 to INT32_MAX)
        if self.sample_full_distribution: 
            if self.seed_buffer_size > 0:
                if self.replay_schedule == 'fixed':
                    if self._proportion_filled >= self.rho and np.random.rand() < self.replay_prob:
                        return self._sample_replay_level()
                    else:
                        return self._sample_unseen_level()
                else:
                    raise ValueError(f'Unsupported replay schedule {self.replay_schedule} when sample_full_distribution=True.')
            else:
                # If seed buffer has length 0, then just sample new random seed each time
                return int(np.random.randint(1,INT32_MAX))

        if strategy == 'random':
            seed_idx = np.random.choice(range((len(self.seeds))))
            seed = self.seeds[seed_idx]
            return int(seed)

        if strategy == 'sequential':
            seed_idx = self.next_seed_index
            self.next_seed_index = (self.next_seed_index + 1) % len(self.seeds)
            seed = self.seeds[seed_idx]
            return int(seed)

        num_unseen = (self.unseen_seed_weights > 0).sum()
        proportion_seen = (len(self.seeds) - num_unseen)/len(self.seeds)

        if self.replay_schedule == 'fixed':
            if proportion_seen >= self.rho: 
                # Sample replay level with fixed replay_prob OR if all levels seen
                if np.random.rand() < self.replay_prob or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else: # Default to proportionate schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay_level()
            else:
                return self._sample_unseen_level()

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1-self.unseen_seed_weights) # Zero out unseen levels

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1-self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0: 
                staleness_weights /= z
            elif z == 0:
                staleness_weights = 1./len(staleness_weights)

            weights = (1 - self.staleness_coef)*weights + self.staleness_coef*staleness_weights

        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            weights = np.ones_like(scores)
        if transform == 'max':
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -float('inf') # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.
        elif transform == 'eps_greedy':
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1. - self.eps
            weights += self.eps/len(self.seeds)
        elif transform == 'rank':
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1./temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores)/temperature)
        elif transform == 'match':
            weights = np.array([(1-score)*score for score in scores])
            weights = weights ** (1./temperature)
        elif transform == 'match_rank':
            weights = np.array([(1-score)*score for score in scores])
            temp = np.flip(weights.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)

        return weights

