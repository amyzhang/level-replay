import sys 
import random 
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class UCBDrAC():
    """
    Upper Confidence Bound Data-regularized Actor-Critic (UCB-DrAC) object
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 aug_list=None,
                 aug_id=None,
                 aug_coef=0.1,
                 num_aug_types=8,
                 ucb_exploration_coef=0.5,
                 ucb_window_length=10,
                 mix_alpha=None,
                 log_grad_norm=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
            
        self.aug_list = aug_list
        self.aug_id = aug_id
        self.aug_coef = aug_coef

        self.ucb_exploration_coef = ucb_exploration_coef
        self.ucb_window_length = ucb_window_length

        self.num_aug_types = num_aug_types
        self.total_num = 1
        self.num_action = [1.] * self.num_aug_types 
        self.qval_action = [0.] * self.num_aug_types 

        self.expl_action = [0.] * self.num_aug_types 
        self.ucb_action = [0.] * self.num_aug_types 
    
        self.return_action = []
        for i in range(num_aug_types):
            self.return_action.append(deque(maxlen=ucb_window_length))

        self.step = 0

        self.mix_alpha = mix_alpha

        self.log_grad_norm = log_grad_norm

    def _grad_norm(self):
        total_norm = 0
        for p in self.actor_critic.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def bandit_state_dict(self):
        return {
            'total_num':self.total_num,
            'num_action':self.num_action,
            'qval_action':self.qval_action,
            'return_action':self.return_action,
            'current_aug_id':self.current_aug_id
        }

    def load_bandit_state_dict(self, state_dict):
        self.total_num = state_dict['total_num']
        self.num_action = state_dict['num_action']
        self.qval_action = state_dict['qval_action']
        self.return_action = state_dict['return_action']
        self.current_aug_id = state_dict['current_aug_id']
        self.current_aug_func = self.aug_list[self.current_aug_id]

    def select_ucb_aug(self):
        for i in range(self.num_aug_types):
            self.expl_action[i] = self.ucb_exploration_coef * \
                np.sqrt(np.log(self.total_num) / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]
        ucb_aug_id = np.argmax(self.ucb_action)
        return ucb_aug_id, self.aug_list[ucb_aug_id]

    def update_ucb_values(self, rollouts):
        self.total_num += 1
        self.num_action[self.current_aug_id] += 1
        self.return_action[self.current_aug_id].append(rollouts.returns.mean().item())
        self.qval_action[self.current_aug_id] = np.mean(self.return_action[self.current_aug_id])
        
    def update(self, rollouts):
        self.step += 1
        self.current_aug_id, self.current_aug_func = self.select_ucb_aug() 

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        if self.log_grad_norm:
            grad_norms = [] 

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                if self.mix_alpha is not None:
                    # Interpolate states and supervision
                    batch_size = len(obs_batch)
                    coeff = np.random.beta(self.mix_alpha, self.mix_alpha, size=(batch_size,))
                    seq_indices = np.arange(batch_size)
                    rand_indices = np.random.permutation(batch_size)
                    indices = np.where(coeff > 0.5, seq_indices, rand_indices)
                    other_indices = np.where(coeff > 0.5, rand_indices, seq_indices)
                    coeff = np.where(coeff > 0.5, coeff, 1 - coeff)
                    coeff = torch.tensor(coeff, dtype=float, device=self.actor_critic.device)

                    old_action_log_probs_batch = coeff*old_action_log_probs_batch[indices] + (1-coeff)*old_action_log_probs_batch[other_indices]
                    return_batch = coeff*return_batch[indices] + (1-coeff)*return_batch[other_indices]
                    values = coeff*values[indices] + (1-coeff)*values[other_indices]
                    adv_targ = coeff*adv_targ[indices] + (1-coeff)*adv_targ[other_indices]
                    action_log_probs = action_log_probs[indices]
                    
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                
                obs_batch_aug = self.current_aug_func.do_augmentation(obs_batch)
                obs_batch_id = self.aug_id(obs_batch)
                
                _, new_actions_batch, _, _ = self.actor_critic.act(\
                    obs_batch_id, recurrent_hidden_states_batch, masks_batch)
                values_aug, action_log_probs_aug, dist_entropy_aug, _ = \
                    self.actor_critic.evaluate_actions(obs_batch_aug, \
                    recurrent_hidden_states_batch, masks_batch, new_actions_batch)
                
                # Compute Augmented Loss
                action_loss_aug = - action_log_probs_aug.mean()
                value_loss_aug = .5 * (torch.detach(values) - values_aug).pow(2).mean()

                # Update actor-critic using both PPO and Augmented Loss
                self.optimizer.zero_grad()
                aug_loss = value_loss_aug + action_loss_aug
                (value_loss * self.value_loss_coef + action_loss -
                    dist_entropy * self.entropy_coef + 
                    aug_loss * self.aug_coef).backward()

                if self.log_grad_norm:
                    grad_norms.append(self._grad_norm())

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()  
                    
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                self.current_aug_func.change_randomization_params_all()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        info = {}
        if self.log_grad_norm:
            info = {'grad_norms': grad_norms}

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, info
