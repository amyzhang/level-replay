import argparse, glob, fnmatch, os, csv, json, re
from collections import defaultdict

import numpy as np

from test_with_lengths import evaluate_saved_model
from level_replay.utils import DotDict

"""
Example usage: 

python evaluate_models.py \
--verbose \
--env_type procgen \
-b ~/logs/ppo/random \
--xpid_template \
lr-ppo-ENV_NAME-random-level_sampler-large-25m-s200
"""

PROCGEN_ENVS = [
	"bigfish",
	"bossfight",
	"caveflyer",
	"chaser",
	"climber",
	"coinrun",
	"dodgeball",
	"fruitbot",
	"heist",
	"jumper",
	"leaper",
	"maze",
	"miner",
	"ninja",
	"plunder",
	"starpilot",
]

MINIGRID_ENVS = [
	"MiniGrid-MultiRoom-N4-S7-Random-v0",
	"MiniGrid-ObstructedMazeGamut-Easy-v0",
	"MiniGrid-ObstructedMazeGamut-Medium-v0"
]


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
	'--xpid', 
	type=str, 
	default='')

	parser.add_argument(
	'-b', '--base_path', 
	type=str, 
	default='~/logs/ppo')

	parser.add_argument(
	'--xpid_template', 
	type=str, 
	default='')

	parser.add_argument(
	'-o', '--output_dir', 
	type=str, 
	default='results')

	parser.add_argument(
	'--env_type', 
	type=str,
	default='procgen',
	choices=['procgen', 'minigrid'], 
	action="store_true") 

	parser.add_argument(
	'--verbose', 
	action="store_true") 

	args = parser.parse_args()

	return args

def main():
	args = parse_args()
	
	base_path = os.path.expandvars(os.path.expanduser(args.base_path))

	if not os.path.exists(args.output_dir):
		print("Creating log directory: %s", args.output_dir)
		os.makedirs(args.output_dir, exist_ok=True)

	out_path = os.path.join(args.output_dir, args.xpid_template + '.csv')
	is_new_file = not os.path.exists(out_path)
	fout = open(out_path, 'a')
	csvwriter = csv.DictWriter(fout, ['env_name', 'mean_episode_length', 'episode_returns'])
	if is_new_file:
		csvwriter.writeheader()

	ENV_NAMES = []
	if args.env_type == 'procgen':
		ENV_NAMES = PROCGEN_ENVS
	elif args.env_type == 'minigrid':
		ENV_NAMES = MINIGRID_ENVS

	for env_name in ENV_NAMES:
		pattern = re.sub(r'ENV_NAME', env_name, args.xpid_template) + '*'
		xpids = fnmatch.filter(os.listdir(base_path), pattern)

		info = defaultdict(list)
		for xpid in xpids:
			# Load meta.json into flags object
			xpid_dir = os.path.join(base_path, xpid)
			meta_json_path = os.path.join(xpid_dir, 'meta.json')
			model_path = os.path.join(xpid_dir, 'model.tar')
			if os.path.exists(model_path):
				print(meta_json_path)			
				with open(meta_json_path) as meta_json_file:
					flags = DotDict(json.load(meta_json_file)['args'])

				episode_returns, mean_episode_length = evaluate_saved_model(
					flags, base_path, xpid, 
					num_episodes=100,
					progressbar=args.verbose, include_episode_length=True, verbose=args.verbose, aggregate_results=False)

				info['episode_returns'] = info['episode_returns'] + episode_returns
				info['mean_episode_length'].append(mean_episode_length)

		episode_returns = ', '.join([str(r) for r in info['episode_returns']])
		mean_episode_length = np.mean(info['mean_episode_length'])

		# Log averages
		csvwriter.writerow({'env_name': env_name, 'mean_episode_length': mean_episode_length, 'episode_returns': episode_returns})
		fout.flush()

if __name__ == '__main__':
	main()
	