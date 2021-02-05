import csv, json, os, argparse, fnmatch
from collections import defaultdict

import pandas as pd
import scipy.stats
import numpy as np

"""
Usage:

python ppo_norm.py \
--sigfigs 1 \
-b results/procgen \
-f \
lr-ppo-ENV_NAME-random-level_sampler-large-25m-s200.csv \
lr-ppo-track_grad-ENV_NAME-K10-alpha1.0-tscl_window-eps_greedy-tau1.0-eps0.5s200.csv \
lr-mixreg-alpha0.01-ENV_NAME-random-s200.csv \
lr-ppo-ucb-ENV_NAME-random-rank-tau1.0-s200.csv \
lr-ppo-plr-ENV_NAME-value_l1-rank-tau0.1-staleness0.1_power_temp1.0-s200.csv \
lr-ppo-ucb-ENV_NAME-value_l1-rank-tau0.1-staleness0.1_power_temp1.0-s200.csv \
-l \
Uniform \
TSCL \
mixreg \
UCB-DrAC \
PLR \
'UCB-DrAC + PLR' \
--compare_max \
--mode latex
"""
CAP_ENV_NAMES = {
	"bigfish": "BigFish",
	"bossfight": "BossFight",
	"caveflyer": "CaveFlyer",
	"chaser": "Chaser",
	"climber": "Climber",
	"coinrun": "CoinRun",
	"dodgeball": "Dodgeball",
	"fruitbot": "FruitBot",
	"heist": "Heist",
	"jumper": "Jumper",
	"leaper": "Leaper",
	"maze": "Maze",
	"miner": "Miner",
	"ninja": "Ninja",
	"plunder": "Plunder",
	"starpilot": "StarPilot"
}

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
	'-b', '--base_path', 
	type=str, default='results')

	parser.add_argument(
	'-f', '--files', 
	type=str, nargs='+')

	parser.add_argument(
	'-l', '--labels', 
	type=str, nargs='+',
	default=[])

	parser.add_argument(
	'--mode',
	type=str, choices=['latex', 'markdown'],
	default='latex'
	)

	parser.add_argument(
	'--show_mean',
	action='store_true'
	)

	parser.add_argument(
	'--show_median',
	action='store_true'
	)

	parser.add_argument(
	'--show_seed_mean',
	action='store_true'
	)

	parser.add_argument(
	'--show_seed_median',
	action='store_true'
	)

	parser.add_argument(
	'-n', '--num_episodes', 
	type=int,
	default=100)

	parser.add_argument(
	'--sigfigs', 
	type=int,
	default=2)

	parser.add_argument(
	'-p', '--pvalue', 
	type=float,
	default=0.05)

	parser.add_argument(
	'--mark_sig', 
	action='store_true')

	parser.add_argument(
	'--compare_max', 
	action='store_true')

	args = parser.parse_args()

	return args

def print_table_header(mode, labels):
	subheader_tokens = ['|']
	if mode == 'markdown':
		tokens = ['|']
		for l in labels:
			tokens.append(f'{l}|')
		header = ''.join(tokens)
		print('|Environment' + header)

		subheader = '|' + '------|'*(len(labels) + 1)
		print(subheader)
	else: 
		tokens = []
		for i, l in enumerate(labels):
			if i == len(labels) - 1:
				tokens.append(f'{l}')
			else:
				tokens.append(f'{l} &')
		header = ''.join(tokens)
		print('Environment &' + header + '\\\\')
		print('\\midrule')

def print_table_contents(
	file2results, 
	labels,
	mode,
	show_mean=True, 
	show_median=False, 
	show_seed_mean=True, 
	show_seed_median=True,
	pvalue=0.05,
	mark_sig=True,
	compare_max=False):
	print(f'{mode} table contents:')

	print_table_header(mode, labels)
	print_table_rows(
		file2results, 
		mode,
		show_mean=show_mean, 
		show_median=show_median, 
		show_seed_mean=show_seed_mean, 
		show_seed_median=show_seed_median,
		pvalue=pvalue,
		mark_sig=mark_sig,
		compare_max=compare_max)

def print_table_rows(
	file2results, 
	mode,
	show_mean=True, 
	show_median=False, 
	show_seed_mean=True, 
	show_seed_median=True,
	pvalue=0.05,
	mark_sig=True,
	compare_max=False):
	if mode == 'markdown':
		plusminus = '&pm;'
		separator = '|'
	else:
		mode = 'latex'
		plusminus = '\\pm'
		separator = '&'

	env_names = file2results[list(file2results.keys())[0]].keys()
	for i, env_name in enumerate(env_names):
		tokens = []
		if mode == 'markdown':
			tokens.append(separator)
		legible_env_name = CAP_ENV_NAMES.get(env_name, env_name)
		if env_name == 'ppo_norm_stats':
			percent = mode == 'markdown' and '%' or '\\%'
			legible_env_name = f'Normalized Returns ({percent})'
		tokens.append(f'{legible_env_name}{separator}')

		sigfigs = args.sigfigs
		if env_name == 'ppo_norm_stats' and mode == 'latex':
			show_mean = show_seed_mean
			show_median = show_seed_median
			sigfigs = 1
			print('\\midrule')

		means = defaultdict(list)
		medians = defaultdict(list)
		returns = defaultdict(list)
		for f, results in file2results.items():
			stats = results[env_name]
			mean = round(stats['mean'], sigfigs)
			std = round(stats['std'], sigfigs)
			stats = results[env_name]
			means[env_name].append(mean)
			medians[env_name].append(std)

			returns[env_name].append(stats['returns'])

		base_returns = pd.DataFrame(data=returns[env_name][0])

		# get max returns
		max_returns = None
		if compare_max:
			max_return_value = float('-inf')
			for j, (f, results) in enumerate(file2results.items()):
				result_mean = results[env_name]['mean']
				if result_mean > max_return_value:
					max_return_value = result_mean
					max_returns = results[env_name]['returns']

		for j, (f, results) in enumerate(file2results.items()):
			stats = results[env_name]

			mean = round(stats['mean'], sigfigs)
			std = round(stats['std'], sigfigs)
			median = round(stats['median'], sigfigs)
			returns = stats['returns']

			method_returns = pd.DataFrame(data=returns)
			comparison_returns = base_returns
			
			if compare_max:
				comparison_returns = max_returns

			sigtest_results = scipy.stats.ttest_ind(comparison_returns, method_returns, equal_var=False, axis=0)

			if compare_max:
				sigtest_success = not (sigtest_results.pvalue[0] < pvalue)
			else:
				sigtest_success = sigtest_results.pvalue[0] < pvalue

			is_max = mean == np.max(means[env_name])
			if show_mean and show_median:
				cell_value = f'{mean}{plusminus}{std} ({median})'

			elif show_median:
				is_max = median == np.max(medians[env_name])
				cell_value = f'{median}'
			else:
				cell_value = f'{mean}{plusminus}{std}'

			if sigtest_success and mark_sig:
				cell_value = f'{cell_value}*'

			if (not compare_max and is_max) or (compare_max and sigtest_success):
				if mode == 'markdown':
					cell_value = f'**{cell_value}**'
				else:
					cell_value = f'\\mathbf{{{cell_value}}}'

			if mode == 'latex':
				cell_value = f'${cell_value}$'

			tokens.append(f'{cell_value}')
			if j < len(file2results) - 1:
				tokens.append(separator)

		if mode == 'markdown':
			tokens.append(separator)
		else:
			tokens.append('\\\\')

		print(''.join(tokens))

if __name__ == '__main__':
	args = parse_args()

	base_path = os.path.expandvars(os.path.expanduser(args.base_path))
	if args.files == '*':
		files = fnmatch.filter(os.listdir(base_path), args.files)
	else:
		files = [os.path.join(base_path, f) for f in args.files]

	next_label_idx = len(args.labels) - 1
	while len(args.labels) < len(files):
		args.labels.append(os.path.basename(files[next_label_index]))
		next_label_idx += 1

	# Compute normalization stats
	ppo_fname = files[0]
	ppo_means = {}
	ppo_norm_std = {}
	with open(ppo_fname) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			env_name = row['env_name']
			returns = row['episode_returns']
			ppo_means[env_name] = np.array([float(v) for v in returns.strip("'").split(', ')]).mean()


	file2results = defaultdict(dict)
	for i, f in enumerate(files):
		print(f'{os.path.basename(f)}:')

		num_episodes_per_seed = args.num_episodes

		ppo_norm_mean_returns = {}
		with open(f) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				env_name = row['env_name']
				returns = row['episode_returns']
				returns = np.array([float(v) for v in returns.strip("'").split(', ')])

				returns = returns.reshape(-1, num_episodes_per_seed).mean(1)

				ppo_norm_mean_returns[env_name] = returns/ppo_means[env_name]
				ppo_norm_std[env_name] = ppo_norm_mean_returns[env_name].std()

				file2results[f][env_name] = {
					'mean': returns.mean(),
					'median': np.median(returns),
					'std': returns.std(),
					'returns': returns
				}
				print(f'{env_name:24}: {returns.mean()} +/- {returns.std()}')

		# Std of mean performance across seeds
		num_seeds = len(returns)
		per_seed_gains = []
		for seed in range(num_seeds):
			per_seed_gains.append([v[seed] for _, v in ppo_norm_mean_returns.items()])

		per_seed_avg_gains = np.array([np.mean(g) for g in per_seed_gains])

		seed_mean = per_seed_avg_gains.mean()
		seed_median = np.median(per_seed_avg_gains)
		seed_std = per_seed_avg_gains.std()

		print(f'Statistics across {num_seeds} seeds:')
		print(f'mean: {seed_mean}')
		print(f'median: {seed_median}')
		print(f'std: {seed_std}')

		file2results[f]['ppo_norm_stats'] = {
			'mean': seed_mean*100,
			'median': seed_median*100,
			'std': seed_std*100,
			'returns': per_seed_avg_gains
		}

	# Print aggregate table across all methods
	print('\n')
	print_table_contents(
		file2results, 
		labels=args.labels,
		mode=args.mode,
		show_mean=args.show_mean, 
		show_median=args.show_median,
		show_seed_mean=args.show_seed_mean,
		show_seed_median=args.show_seed_median,
		pvalue=args.pvalue,
		mark_sig=args.mark_sig,
		compare_max=args.compare_max)


