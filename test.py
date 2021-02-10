import sys
import os
import logging

import numpy as np
import torch
from tqdm import tqdm

from level_replay import utils
from level_replay.model import model_for_env_name
from level_replay.level_sampler import LevelSampler

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecNormalize
)
from level_replay.envs import make_lr_venv

INT32_MAX = 2147483647


def evaluate(
    args, 
    actor_critic, 
    num_episodes, 
    device, 
    num_processes=1, 
    deterministic=False,
    start_level=0,
    num_levels=0,
    seeds=None,
    level_sampler=None, 
    progressbar=None,
    aug_id=None,
    include_episode_length=False):
    actor_critic.eval()
        
    if level_sampler:
        start_level = level_sampler.seed_range()[0]
        num_levels = 1

    eval_envs, level_sampler, _ = make_lr_venv(
        num_envs=num_processes, env_name=args.env_name,
        seeds=seeds, device=device,
        num_levels=num_levels, start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler=level_sampler)

    eval_episode_rewards = []

    if include_episode_length:
        eval_episode_lengths = []

    if level_sampler:
        obs, _ = eval_envs.reset()
    else:
        obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.ones(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            if aug_id:
                obs = aug_id(obs)
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        obs, _, done, infos = eval_envs.step(action)
         
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                if include_episode_length:
                    eval_episode_lengths.append(info['episode']['l'])
                if progressbar:
                    progressbar.update(1)

    eval_envs.close()
    if progressbar:
        progressbar.close()

    if args.verbose:
        print("Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n"\
            .format(len(eval_episode_rewards), \
            np.mean(eval_episode_rewards), np.median(eval_episode_rewards)))

    if include_episode_length:
        return eval_episode_rewards, eval_episode_lengths

    return eval_episode_rewards


def evaluate_saved_model(
    args,
    result_dir,
    xpid, 
    num_episodes=10, 
    seeds=None, 
    deterministic=False, 
    verbose=False, 
    progressbar=False,
    num_processes=1,
    include_episode_length=False):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')

    if verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.xpid is None:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(os.path.join(result_dir, "latest", "model.tar"))
        )
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(os.path.join(result_dir, xpid, "model.tar"))
        )

    # Set up level sampler
    if seeds is None:
        seeds = [int(np.random.randint(1,INT32_MAX)) for _ in range(num_episodes)]

    dummy_env, _ = make_lr_venv(
        num_envs=num_processes, env_name=args.env_name,
        seeds=None, device=device,
        num_levels=1, start_level=1,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info)

    level_sampler = LevelSampler(
        seeds, 
        dummy_env.observation_space, dummy_env.action_space,
        strategy='sequential')

    model = model_for_env_name(args, dummy_env)

    pbar = None
    if progressbar:
        pbar = tqdm(total=num_episodes)

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(checkpointpath, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    num_processes = min(num_processes, num_episodes)
        
    if include_episode_length:
        eval_episode_rewards, eval_episode_lengths = \
            evaluate(args, model, num_episodes, 
                device=device, 
                num_processes=num_processes, 
                level_sampler=level_sampler, 
                progressbar=pbar,
                include_episode_length=include_episode_length)
    else:
        eval_episode_rewards = \
            evaluate(args, model, num_episodes, 
                device=device, 
                num_processes=num_processes, 
                level_sampler=level_sampler, 
                progressbar=pbar,
                include_episode_length=include_episode_length)

    mean_return = np.mean(eval_episode_rewards)
    median_return = np.median(eval_episode_rewards)

    if include_episode_length:
        mean_episode_length = np.mean(eval_episode_lengths)

    logging.info(
        "Average returns over %i episodes: %.2f", num_episodes, mean_return
    )
    logging.info(
        "Median returns over %i episodes: %.2f", num_episodes, median_return
    )

    if include_episode_length:
        logging.info(
            "Average episode length over %i episodes: %.2f", num_episodes, mean_episode_length 
        )

    if include_episode_length:
        return mean_return, median_return, mean_episode_length

    return mean_return, median_return
