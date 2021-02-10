import argparse

import torch

from .utils import str2bool

parser = argparse.ArgumentParser(description='RL')

# PPO Arguments.
parser.add_argument(
    '--lr', 
    type=float, 
    default=5e-4, 
    help='learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discount factor for rewards')
parser.add_argument(
    '--use_gae',
    type=str2bool, nargs='?', const=True, default=True,
    help='Use generalized advantage estimator.')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='gae lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.01,
    help='entropy term coefficient')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.5,
    help='value loss coefficient (default: 0.5)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max norm of gradients)')
parser.add_argument(
    '--ensemble_size',
    type=int,
    default=5,
    help='ensemble size')
parser.add_argument(
    '--no_ret_normalization',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to use unnormalized returns')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='random seed')
parser.add_argument(
    '--num_processes',
    type=int,
    default=64,
    help='how many training CPU processes to use')
parser.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='number of forward steps in A2C')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=3,
    help='number of ppo epochs')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=8,
    help='number of batches for ppo')
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='ppo clip parameter')
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=25e6,
    help='number of environment steps to train')
parser.add_argument(
    '--env_name',
    type=str,
    default='bigfish',
    help='environment to train on')
parser.add_argument(
    '--xpid',
    default='latest',
    help='name for the run - prefix to log files')
parser.add_argument(
    '--log_dir',
    # default='/checkpoint/amyzhang/level-replay/',
    default='~/logs/ppo',
    help='directory to save agent logs')
parser.add_argument(
    '--no_cuda',
    type=str2bool, nargs='?', const=True, default=False,
    help='disables CUDA training')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256,
    help='state embedding dimension')
parser.add_argument(
    '--arch',
    type=str,
    default='large',
    choices=['small', 'large'],
    help='agent architecture')
parser.add_argument(
    '--algo',
    type=str,
    default='ppo',
    choices=['ppo', 'a2c', 'acktr', 'ucb', 'mixreg'],
    help='Which RL algorithm to use.')
parser.add_argument(
    '--kl_clip',
    type=float,
    default=0.001,
    help='Trust-region radius for ACKTR')

# Procgen arguments.
parser.add_argument(
    '--distribution_mode',
    default='easy',
    help='distribution of envs for procgen')
parser.add_argument(
    '--paint_vel_info',
    type=str2bool, nargs='?', const=True, default=False,
    help='Paint velocity vector at top of frames.')
parser.add_argument(
    '--num_train_seeds',
    type=int,
    default=200,
    help='number of Procgen levels to use for training')
parser.add_argument(
    '--start_level',
    type=int,
    default=0,
    help='start level id for sampling Procgen levels')
parser.add_argument(
    "--num_test_seeds", 
    type=int,
    default=10,
    help="Number of test seeds")
parser.add_argument(
    "--final_num_test_seeds", 
    type=int,
    default=1000,
    help="Number of test seeds")
parser.add_argument(
    '--seed_path',
    type=str,
    default=None,
    help='Path to file containing specific training seeds')
parser.add_argument(
    '--use_env_name_seed_file',
    type=str2bool, nargs='?', const=True, default=False,
    help='Whether to load seeds from file starting with env name in seeds/')

# Level Replay arguments.
parser.add_argument(
    "--level_replay_score_transform",
    type=str, 
    default='softmax', 
    choices=['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_temperature", 
    type=float,
    default=1.0,
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_strategy", 
    type=str,
    default='random',
    choices=['off', 'random', 'sequential',
            'policy_entropy', 'least_confidence', 'min_margin', 'gae', 'value_l1', 'one_step_td_error',
            'tscl_window', 'uncertainty'],
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_max_score_coef", 
    type=float,
    default=0.0,
    help="How much to weigh max vs. average for trajectory scores")
parser.add_argument(
    "--level_replay_eps", 
    type=float,
    default=0.05,
    help="Level replay epsilon for eps-greedy sampling")
parser.add_argument(
    "--level_replay_schedule",
    type=str,
    default='proportionate',
    help="Level replay schedule for sampling seen levels")
parser.add_argument(
    "--level_replay_rho",
    type=float, 
    default=1.0,
    help="Minimum size of replay set relative to total number of levels before sampling replays.")
parser.add_argument(
    "--level_replay_prob", 
    type=float,
    default=0.95,
    help="Probability of sampling a new level instead of a replay level.")
parser.add_argument(
    "--level_replay_alpha",
    type=float, 
    default=1.0,
    help="Level score EWA smoothing factor")
parser.add_argument(
    "--staleness_coef",
    type=float, 
    default=0.0,
    help="Staleness weighing")
parser.add_argument(
    "--staleness_transform",
    type=str, 
    default='power',
    choices=['max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Staleness normalization transform")
parser.add_argument(
    "--staleness_temperature",
    type=float, 
    default=1.0,
    help="Staleness normalization temperature.")

# Secondary strategy arguments.
parser.add_argument(
    "--level_replay_secondary_strategy", 
    type=str,
    default=None,
    choices=['off', 'random', 'sequential',
            'policy_entropy', 'least_confidence', 'min_margin', 'gae', 'value_l1', 'one_step_td_error',
            'tscl_window', 'uncertainty'],
    help="Level replay secondary scoring strategy")
parser.add_argument(
    "--level_replay_strategy_mix_coef", 
    type=float,
    default=0.5,
    help="Weight to assign primary strategy when mixing with a secondary strategy")
parser.add_argument(
    "--level_replay_secondary_score_transform",
    type=str, 
    default='softmax', 
    choices=['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_secondary_temperature", 
    type=float,
    default=1.0,
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_secondary_eps", 
    type=float,
    default=0.05,
    help="Level replay epsilon for eps-greedy sampling")
parser.add_argument(
    "--secondary_staleness_coef",
    type=float, 
    default=0.0,
    help="Staleness weighing")
parser.add_argument(
    "--secondary_staleness_transform",
    type=str, 
    default='power',
    choices=['max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Staleness normalization transform")
parser.add_argument(
    "--secondary_staleness_temperature",
    type=float, 
    default=1.0,
    help="Staleness normalization temperature")


parser.add_argument(
    "--train_full_distribution",
    type=str2bool, nargs='?', const=True, default=False,
    help='Train on the full distribution of levels.'
)
parser.add_argument(
    "--level_replay_seed_buffer_size",
    type=int, 
    default=0,
    help="Size of seed buffer, a min-priority queue.")
parser.add_argument(
    "--level_replay_seed_buffer_priority",
    type=str, 
    default='replay_support',
    choices=['score', 'replay_support'], 
    help="How to prioritize seed buffer indices.")

# Logging arguments.
parser.add_argument(
    "--verbose", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Whether to print logs")
parser.add_argument(
    '--log_interval',
    type=int,
    default=1,
    help='log interval, one log per n updates')
parser.add_argument(
    "--save_interval", 
    type=int, 
    default=20,
    help="Save model every this many minutes.")
parser.add_argument(
    "--weight_log_interval", 
    type=int, 
    default=1,
    help="Save level weights every this many updates")
parser.add_argument(
    "--checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Begin training from checkpoint.")
parser.add_argument(
    "--disable_checkpoint", 
    type=str2bool, nargs='?', const=True, default=False,
    help="Disable saving checkpoint.")
parser.add_argument(
    '--log_grad_norm',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log the gradient norm of the actor critic")


# Auto-DrAC arguments.
parser.add_argument(
    '--aug_type',
    type=str,
    default='crop',
    help='augmentation type')
parser.add_argument(
    '--aug_coef', 
    type=float, 
    default=0.1, 
    help='coefficient on the augmented loss')
parser.add_argument(
    '--aug_extra_shape', 
    type=int, 
    default=0, 
    help='increase image size by')
parser.add_argument(
    '--image_pad', 
    type=int, 
    default=12, 
    help='increase image size by')

# UCB-DrAC arguments.
parser.add_argument(
    '--use_ucb',
    type=str2bool, nargs='?', const=True, default=False,
    help='use UCB to select an augmentation')
parser.add_argument(
    '--ucb_window_length', 
    type=int, 
    default=10, 
    help='length of sliding window for UCB (i.e. number of UCB actions)')
parser.add_argument(
    '--ucb_exploration_coef', 
    type=float, 
    default=0.1, 
    help='exploration coefficient for UCB')

# TSCL arguments.
parser.add_argument(
    "--tscl_window_size",
    type=int, default=0,
    help='Window size to use for tscl')

# Mixreg arguments.
parser.add_argument(
    '--use_mixreg',
    type=str2bool, nargs='?', const=True, default=False,
    help='use mixreg')

parser.add_argument(
    '--mixreg_alpha', 
    type=float, 
    default=0.2, 
    help='MixReg Beta-dist alpha')
