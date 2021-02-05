import argparse
import csv
import os
import math

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
mpl.style.use('seaborn')

import seaborn as sns
import gym
from gym_minigrid.wrappers import *
from custom_envs import ObstructedMazeGamut

"""
Usage:

python analyze_curriculum.py \
	--resultdir '~/logs/ppo/minigrid/multiroom-n4-random/' \
	--env_name "MiniGrid-MultiRoom-N4-S7-Random-v0" \
	--xpid "lr-ppo-small-nonorm-MiniGrid-MultiRoom-N4-S7-Random-v0-value_l1-staleness0.3-300m-s4000_0" \
	--xtick 1 \
	--xinc 1
"""
def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-r", "--resultdir", type=str, default="~/logs/ppo/")
	parser.add_argument("--xpid", type=str, default=None)
	parser.add_argument("--env_name", type=str, default="MiniGrid-MultiRoom-N4-S7-Random-v0")
	parser.add_argument("--xtick", type=int, default=1, help="Plot every this many updates. This is the bin size in terms of # rows in barchart mode.")
	parser.add_argument("--xinc", type=int, default=1, help="Number of increments represented per x tick.")
	parser.add_argument("--max", action='store_true')
	parser.add_argument("--xlabel", type=str, default='Average level proportions')
	parser.add_argument("--ylabel", type=str, default='Steps')

	parser.add_argument("--barchart", action='store_true')
	parser.add_argument("--bar_width", type=int, default=100)
	parser.add_argument("--max_row", type=int, default=None)
	parser.add_argument("--alpha", type=float, default=0.1)
	parser.add_argument("--savepath", type=str, default='figures/')
	parser.add_argument("--savename", type=str, default=None)
	parser.add_argument("--fontsize", type=int, default=6)

	args = parser.parse_args()

	return args

def reformat_large_tick_values(tick_val, pos=None):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    
    From: https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)      
    elif tick_val >= 10:
        new_tick_format = round(tick_val, 1)
    elif tick_val >= 1:
        new_tick_format = round(tick_val, 2)        
    elif tick_val >= 1e-4:
        new_tick_format = round(tick_val, 3)
    elif tick_val >= 1e-8:         
        new_tick_format = round(tick_val, 8)   
    else:
        new_tick_format = tick_val

    new_tick_format = str(new_tick_format)
    new_tick_format = new_tick_format if "e" in new_tick_format else new_tick_format[:6]
    index_of_decimal = new_tick_format.find(".")
    
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0" and (tick_val >= 10 or tick_val <= -10 or tick_val == 0.0):
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]
            
    # Manual hack
    if new_tick_format == "-0.019":
        new_tick_format = "-0.02"
    elif new_tick_format == "-0.039":
        new_tick_format = "-0.04"
            
    return new_tick_format

def round_to_integer_parts(percents, total):
	floor_parts = [math.floor(p*total) for p in percents]
	diff = total - np.sum(floor_parts)
	sorted_i = np.argsort([x - math.floor(x) for x in percents])[::-1]
	inverse_sorted_i = np.argsort(sorted_i)
	floor_parts = np.array(floor_parts, dtype=int)[sorted_i]

	for i in range(diff):
		floor_parts[i] += 1

	return floor_parts[inverse_sorted_i]


def normalized_sum_every_n_rows(a, n):
	n_rows = a.shape[0]
	return np.array([a[k:k+n].sum(0)/len(a[k:k+n]) for k in range(0, n_rows, n)])


def map_env_seeds_to_difficulty(env_name, seeds):
	env = gym.make(env_name)

	seed2difficulty = {}
	max_difficulty = 0
	if env_name.startswith('MiniGrid-MultiRoom'):
		max_difficulty = env.maxNumRooms
	elif env_name.startswith('MiniGrid-ObstructedMaze'):
		max_difficulty = env.max_difficulty

	for i, seed in enumerate(seeds):
		print(f'Checking seed #{i}: {seed}')
		env.seed(seed)
		obs = env.reset()

		if env_name.startswith('MiniGrid-MultiRoom'):
			difficulty = len(env.rooms) - 1
		elif env_name.startswith('MiniGrid-ObstructedMaze'):
			difficulty = seed % env.max_difficulty
		else:
			raise ValueError(f'Unsupported env {env_name}')

		seed2difficulty[seed] = difficulty

	return seed2difficulty, max_difficulty


if __name__ == '__main__':
	"""
	Reads in a seed weight file and maps each seed to a level difficulty and plots a 
	stacked area chart of showing the probability mass for each difficulty.
	"""
	args = parse_args()

	dpi = 100
	width = 220
	plt.figure(
		figsize=(width / dpi, width * (5 ** 0.5 - 1) * 0.6667 / 2 / dpi), dpi=dpi
	)

	resultdir = os.path.expandvars(os.path.expanduser(args.resultdir))
	basepath = os.path.join(resultdir, args.xpid if args.xpid else 'latest')
	filepath = f'{basepath}/level_weights.csv'

	if args.savename:
		savepath = os.path.expandvars(os.path.expanduser(args.savepath))
		savepath = os.path.join(savepath, args.savename)

	if not os.path.exists(filepath):
		raise ValueError(f'File does not exist: {filepath}')

	seeds = []
	# Read in weights
	weights_per_time = []
	f_in = open(filepath, 'r')
	meta = f_in.readline()
	if meta.startswith('#'):
		meta = meta.replace(r'#', '').strip()
		meta = meta.split(',')
		seeds = [int(x) for x in meta] 
		reader = csv.reader(f_in)
	else:
		raise ValueError(f'Missing seeds header in {filepath}.')

	seed2difficulty, max_difficulty = map_env_seeds_to_difficulty(args.env_name, seeds)
	seed_difficulties = np.array([v for _, v in seed2difficulty.items()], dtype=float)

	if args.barchart:
		difficulty2indices = {}
		for difficulty in range(max_difficulty):
			difficulty2indices[difficulty] = np.where(seed_difficulties == difficulty)[0]

		difficulty_counts = []

		for i, row in enumerate(reader):
			if args.max_row and i >= args.max_row:
				break
			if row[0].startswith('#'): continue
			seed_weights = np.array([float(w) for w in row])
			z = seed_weights.sum()
			if z != 1.0: seed_weights /= z
			if args.max:
				max_i = np.argmax(seed_weights)
				seed_weights = np.zeros_like(seed_weights)
				seed_weights[max_i] = 1.0

			row_difficulty_counts = np.zeros(max_difficulty)
			for difficulty in range(max_difficulty):
				indices = difficulty2indices[difficulty]
				row_difficulty_counts[difficulty] = seed_weights[indices].sum()
			difficulty_counts.append(row_difficulty_counts)

		# discretize difficulty counts
		difficulty_counts = np.stack(difficulty_counts, axis=0)

		# Smoothing
		df = pd.DataFrame(data=difficulty_counts)
		difficulty_counts = df.ewm(alpha=args.alpha, ignore_na=True).mean().to_numpy()

		# Binning
		difficulty_counts = normalized_sum_every_n_rows(difficulty_counts, args.xtick)

		# Discretize to integer bars
		difficulty_counts = np.array([round_to_integer_parts(row.tolist(), args.bar_width) for row in difficulty_counts], dtype=int)
		bars = np.zeros((difficulty_counts.shape[0], args.bar_width), dtype=int)
		for i, counts in enumerate(difficulty_counts):
			start = 0
			for difficulty, count in enumerate(counts):
				bars[i, start:start+count] = difficulty
				start += count
		y = bars

		# Plot barchart
		plt.grid(b=None)
		plt.imshow(
			np.flip(np.transpose(y), 0), 
			extent=[0,y.shape[0]*args.xtick*args.xinc,0,1.0], 
			interpolation='nearest', 
			cmap=plt.get_cmap('viridis', max_difficulty), 
			aspect='auto')
		tick_fontsize = args.fontsize

		colorbar = plt.colorbar()
		colorbar.set_ticks(range(max_difficulty))
		colorbar.ax.tick_params(labelsize=args.fontsize)

		ax = plt.gca()
		ax.set_xlabel(args.xlabel, fontsize=args.fontsize)
		ax.set_ylabel(args.ylabel, fontsize=args.fontsize)

		ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
		ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
		ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(reformat_large_tick_values))

	else:
		avg_difficulty = []
		max_difficulty = []
		for i, row in enumerate(reader):
			if row[0].startswith('#'): continue
			if i % args.xtick == 0:
				seed_weights = np.array([float(w) for w in row])

				avg_difficulty_i = (seed_weights*seed_difficulties).mean()
				max_difficulty_i  = seed_difficulties[np.argmax(seed_weights)]

				avg_difficulty.append(avg_difficulty_i)
				max_difficulty.append(max_difficulty_i)

		x = [t*args.xtick*args.xinc for t in range(len(avg_difficulty))]
		y = max_difficulty if args.max else avg_difficulty
		
		plt.xlabel('# updates')
		plt.ylabel('Most likely difficulty' if args.max else 'Average difficulty')

		plt.plot(x, y)

	if args.savename:
		plt.savefig(
			os.path.join(args.savepath, f"{args.savename}.pdf"),
			bbox_inches="tight",
			dpi=dpi,
			pad_inches=0,
		)
	else:
		plt.show()
