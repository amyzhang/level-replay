import argparse, glob, fnmatch, os, csv, json, re
from pathlib import Path
from functools import reduce

import numpy as np
import scipy
from collections import defaultdict
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.style
# mpl.use('TkAgg')
mpl.use("macOSX")
mpl.style.use('seaborn')
import matplotlib.pyplot as plt
import seaborn as sns


def islast(itr):
  old = next(itr)
  for new in itr:
    yield False, old
    old = new
  yield True, old


def file_index_key(f):
	pattern = r'\d+$'
	key_match = re.findall(pattern, Path(f).stem)
	if len(key_match):
		return int(key_match[0])
	return f

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


def gather_results_for_prefix(args, results_path, prefix, point_interval):
	pattern = '{}*'.format(prefix)

	xpids = fnmatch.filter(os.listdir(results_path), pattern)
	xpids.sort(key=file_index_key)

	assert len(xpids) > 0, f'Results for {pattern} not found.'

	pd_series = []

	nfiles = 0
	for i, f in enumerate(xpids):
		f_in = open(os.path.join(results_path, f, args.log_filename), 'rt')
		reader = csv.reader((line.replace('\0', ' ') for line in f_in))
		headers = next(reader, None)
		if len(headers) < 2:
			raise ValueError('result is malformed')
		headers[0] = headers[0].replace('#', '').strip() # remove comment hash and space

		xs = []
		ys = []
		last_x = -1

		for row_index, (is_last, row) in enumerate(islast(reader)):
			if len(row) != len(headers):
				continue

			if args.max_lines and row_index > args.max_lines:
				break
			if row_index % point_interval == 0 or is_last:
				row_dict = dict(zip(headers, row))
				x = int(row_dict[args.x_axis])

				if x < last_x: # Preempted+restarted jobs may have some duplicated x index values
					continue
				last_x = x

				if args.max_x is not None and x > args.max_x:
					break
				if args.gap:
					y = float(row_dict['train_eval:mean_episode_return']) - float(row_dict['test:mean_episode_return'])
				else:
					y = float(row_dict[args.y_axis])

				xs.append(x)
				ys.append(y)

		pd_series.append(pd.Series(ys,index=xs).sort_index(axis=0))
		nfiles += 1

	return nfiles, pd_series

def plot_results_for_prefix(args, ax, results_path, prefix, label, title=None, tag='', is_drac=None):
	if not drac:
		point_interval = args.point_interval
	else:
		point_interval = 10

	nfiles, pd_series = gather_results_for_prefix(args, results_path, prefix, point_interval)

	for i, series in enumerate(pd_series):
		pd_series[i] = series.loc[~series.index.duplicated(keep='first')]
	try:
		df = pd.concat(pd_series, axis=1).interpolate(method='linear')*args.scale
	except:
		df = pd.concat(pd_series, axis=1)*args.scale

	ewm = df.ewm(alpha=args.alpha, ignore_na=True).mean()

	all_x = np.array([i for i in df.index])
	max_x = max(all_x)
	plt_x = all_x
	plt_y_avg = np.array([y for y in ewm.mean(axis=1)])
	plt_y_std = np.array([std for std in ewm.std(axis=1)])

	ax.plot(plt_x, plt_y_avg, linewidth=args.linewidth, label=label)
	ax.fill_between(plt_x, plt_y_avg - plt_y_std, plt_y_avg + plt_y_std, alpha=0.1)

	if title:
		if args.grid:
			ax.set_title(title, fontsize=args.fontsize)
		else:
			ax.title(title, fontsize=args.fontsize)

	info = {'max_x': max_x, 'all_x': all_x, 'avg_y': plt_y_avg, 'std_y': plt_y_std, 'df':ewm, 'tag': tag}
	return info


def format_subplot(args, max_x, all_x, fig, subplt, last_row=False):
	# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3, prop={'size': args.fontsize})
	# tick_fontsize = 4
	tick_fontsize = 6
	subplt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
	subplt.xaxis.get_offset_text().set_fontsize(tick_fontsize)
	subplt.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(reformat_large_tick_values));
	subplt.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mpl.ticker.FormatStrFormatter('%d')));
	subplt.tick_params(axis='y', which='major', pad=-1)
	subplt.tick_params(axis='x', which='major', pad=0)
	subplt.grid(linewidth=0.5)

def format_plot(args, max_x, all_x, fig, plt):
	ax = plt.gca()

	if args.legend_inside:
		fig.legend(loc='lower right', prop={'size': args.fontsize})
	else:
		fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=4, prop={'size': args.fontsize})
		if args.title:
			ax.set_title(args.title, fontsize=8)

	ax.set_xlabel(args.x_label, fontsize=args.fontsize)
	ax.set_ylabel(args.y_label, fontsize=args.fontsize)

	tick_fontsize = args.fontsize
	ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
	ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
	ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(reformat_large_tick_values));

	if args.max_y is not None:
		ax.set_ylim(top=args.max_y)

	if args.min_y is not None:
		ax.set_ylim(bottom=args.min_y)


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

PROCGEN_MIN_MAX_EASY = {
    'coinrun': [5, 10],
    'starpilot': [2.5, 64],
    'caveflyer': [3.5, 12],
    'dodgeball': [1.5, 19],
    'fruitbot': [-1.5, 32.4],
    'chaser': [.5, 13],
    'miner': [1.5, 13],
    'jumper': [1, 10],
    'leaper': [1.5, 10],
    'maze': [5, 10],
    'bigfish': [1, 40],
    'heist': [3.5, 10],
    'climber': [2, 12.6],
    'plunder': [4.5, 30],
    'ninja': [3.5, 10],
    'bossfight': [.5, 13],
}

PROCGEN_MIN_MAX_HARD = {
	"bigfish": (0,40),
	"bossfight": (0.5,13),
	"caveflyer": (2,13.4),
	"chaser": (0.5,14.2),
	"climber": (1,12.6),
	"coinrun": (5,10),
	"dodgeball": (1.5,19),
	"fruitbot": (-0.5,27.2),
	"heist": (2,10),
	"jumper": (1,10),
	"leaper": (1.5,10),
	"maze": (4,10),
	"miner": (1.5,20),
	"ninja": (2,10),
	"plunder": (3,30),
	"starpilot": (1.5,35)
}

if __name__ == '__main__':
	"""
	Arguments:
		--prefix: filename prefix of result files. Results from files with shared filename prefix will be averaged.
		--results_path: path to directory with result files
		--label: labels for each curve
		--max_index: highest index i to consider in computing curves per prefix, where filenames are of form "^(prefix).*_(i)$"
		--alpha: Polyak averaging smoothing coefficient
		--x_axis: csv column name for x-axis data, defaults to "epoch"
		--y_axis: csv column name for y-axis data, defaults to "loss"
		--threshold: show a horizontal line at this y value
		--threshold_label: label for threshold
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--base_path', type=str, default='~/logs/ppo', help='base path to results directory per prefix')
	parser.add_argument('-r', '--results_path', type=str, nargs='+', default=[''], help='path to results directory')
	parser.add_argument('--prefix', type=str, nargs='+', default=[''], help='Plot each xpid group matching this prefix per game')
	parser.add_argument('--log_filename', type=str, default='logs.csv', help='Name of log output file in each result directory')
	parser.add_argument('-lines', '--max_lines', type=int, default=None, help='only plot every this many points')
	parser.add_argument('--grid', action='store_true', help='Plot all prefix tuples per game in a grid')
	parser.add_argument('--xpid_prefix', type=str, nargs='+', default=[], help='Prefix of xpid folders if plotting curves aggregated by subfolders')

	parser.add_argument('--kfwer_ratio', type=float, default=0.5, help='k/N in k-FWER test for significance in differences of curves.')
	parser.add_argument('--pvalue', type=float, default=0.05, help='p-value for significance')

	parser.add_argument('-s', '--scale', type=float, default=1.0, help='scale all values by this constant')
	parser.add_argument('-l', '--label', type=str, nargs='+', default=[None], help='labels')
	parser.add_argument('-m', '--max_index', type=int, help='max index of prefix match to use')

	parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha for emwa')
	parser.add_argument('-x', '--x_axis', type=str, default='step', help='csv column name of x-axis data')
	parser.add_argument('-y', '--y_axis', type=str, default='test:mean_episode_return', help='csv column name of y-axis data')
	parser.add_argument('-yr', '--y_range', type=float, default=[], help='y range')
	parser.add_argument('-xl', '--x_label', type=str, default='Steps', help='x-axis label')
	parser.add_argument('--titles', type=str, nargs='+', default=[''], help='plot titles')
	parser.add_argument('-yl', '--y_label', type=str, default='Mean test episode return', help='y-axis label')
	parser.add_argument('-xi', '--x_increment', type=int, default=1, help='x-axis increment')
	parser.add_argument('-xts', '--x_tick_suffix', type=str, default='M', help='x-axis tick suffix')
	parser.add_argument('-pi', '--point_interval', type=int, default=1, help='only plot every this many points')
	parser.add_argument('--max_x', type=float, default=None, help='max x-value')
	parser.add_argument('--max_y', type=float, default=None, help='max y-value')
	parser.add_argument('--min_y', type=float, default=None, help='max y-value')
	parser.add_argument('--x_values_as_axis', action='store_true', help='Show exactly x-values in data along x-axis')
	parser.add_argument('--ignore_x_values_in_axis', type=float, nargs='+', default=[], help='Ignore these x-values in axis')
	parser.add_argument('--linewidth', type=float, default=1.0, help='line width')
	parser.add_argument('--linestyle', type=str, default='-', help='line style')
	
	parser.add_argument('--threshold', type=float, default=None, help='show a horizontal line at this y value')
	parser.add_argument('--threshold_label', type=str, default='', help='label for threshold')

	parser.add_argument('--save_path', type=str, default='figures/', help='Path to save image')
	parser.add_argument('--savename', type=str, default=None, help='Name of output image')
	parser.add_argument('--dpi', type=int, default=72, help='dpi of saved image')
	parser.add_argument('--save_width', type=int, default=800, help='pixel width of saved image')
	parser.add_argument('--save_height', type=int, default=480, help='pixel height of saved image')
	parser.add_argument('--fontsize', type=int, default=6, help='pixel height of saved image')
	parser.add_argument('--legend_inside', action='store_true', help='show legend inside plot')
	parser.add_argument('--title', type=str, help='title for single plot')

	parser.add_argument('--gap', action="store_true", default=False, help='Whether to plot the generalization gap')
	parser.add_argument('--avg_procgen', action='store_true', help='Average all return-normalized curves')

	parser.add_argument('--env_name', type=str, default=None, help='Specify environment name if plotting individual minigrid results')

	args = parser.parse_args()

	sns.set_style("whitegrid", {"grid.color": "#EFEFEF"})

	# Create an array with the colors you want to use
	# colors = sns.husl_palette(num_colors, h=.1)
	colors = [
		(0.8859561388376407, 0.5226505841897354, 0.195714831410001), # Uniform: orange
		(1., 0.19215686, 0.19215686), # TSCL: red
		(1, 0.7019607843137254, 0.011764705882352941), # mixreg: yellow
 		(0.20964485513246672, 0.6785281560863642, 0.6309437466865638), # UCB-DrAC: Teal
 		(0.9615698478167679, 0.3916890619185551, 0.8268671491444017), # PLR: Pink
		(0.3711152842731098, 0.6174124752499043, 0.9586047646790773), # UCB-DrAC + PLR: Blue
		# (0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green
	]

	# Set your custom color palette
	sns.set_palette(sns.color_palette(colors))

	results_path = args.results_path
	if args.base_path:
		base_path = os.path.expandvars(os.path.expanduser(args.base_path))
		results_path = [os.path.join(base_path, p) for p in results_path]

	if len(args.xpid_prefix) > 0 and len(args.xpid_prefix) < len(results_path):
		args.xpid_prefix = args.xpid_prefix + ([args.xpid_prefix[-1]] * (len(results_path) - len(args.xpid_prefix)))

	label = args.label
	if len(label) < len(results_path):
		label = label + ([''] * (len(results_path) - len(label)))

	titles = args.titles
	if len(titles) < len(results_path):
		titles = titles + ([''] * (len(results_path) - len(titles)))

	dpi = args.dpi
	if args.grid:
		num_plots = len(PROCGEN_ENVS)
		subplot_width = 4
		subplot_height = int(np.ceil(num_plots/subplot_width))
		fig, ax = plt.subplots(subplot_height, subplot_width, sharex='col', sharey=False)
		ax = ax.flatten()
		fig.set_figwidth(args.save_width/dpi)
		fig.set_figheight(args.save_height/dpi)
		fig.set_dpi(dpi)
		plt.subplots_adjust(left=0.025, bottom=0.10, right=0.97, top=.80, wspace=0.05, hspace=0.3)
	else:
		ax = plt
		fig = plt.figure(figsize=(args.save_width/dpi, args.save_height/dpi), dpi=dpi)

	plt_index = 0
	max_x = 0

	results_metas = list(zip(results_path, label))

	print(f'========= Final {args.y_axis} ========')
	infos_dict = {p:[] for p in args.results_path}
	for i, meta in enumerate(results_metas):
		rp, l = results_metas[i]
		print(rp, l)
		drac = False
		tscl = False
		mixreg = False

		if len(args.xpid_prefix) > 0:
			xpid_prefix = args.xpid_prefix[i]
		else:		
			xpid_prefix = 'lr-ppo'
			if l and 'UCB' in l:
				drac = True
			elif l and 'TSCL' in l:
				tscl = True
			elif l and 'mixreg' in l:
				mixreg = True

			if drac:
				xpid_prefix = 'lr-ppo-ucb'
			elif tscl:
				xpid_prefix = 'lr-ppo-track_grad'
			elif mixreg:
				xpid_prefix = 'lr-mixreg-alpha0.01'

		if args.grid:
			l_initial = l
			for j, env in enumerate(PROCGEN_ENVS):
				if j > 0: l = None
				env_p = f"{xpid_prefix}-{env}-"
				info = plot_results_for_prefix(args, ax[j], rp, env_p, l, PROCGEN_ENVS[j], tag=env, is_drac=drac)
				infos_dict[args.results_path[i]].append(info)
				max_x = max(info['max_x'], max_x)

		elif args.avg_procgen:
			all_series = []
			for j, env in enumerate(PROCGEN_ENVS):
				if j > 0: l = None

				env_p = f"{xpid_prefix}-{env}-"
				point_interval = 10 if drac else args.point_interval
				_, pd_series = gather_results_for_prefix(args, rp, env_p, point_interval)

				if not args.gap:
					R_min, R_max = PROCGEN_MIN_MAX_EASY[env]
					pd_series = [p.add(-R_min).divide(R_max - R_min) for p in pd_series]

				all_series.append(pd_series)

			all_series_pd = []
			min_length = float('inf')
			all_series_updated = []
			for series in all_series:
				updated_series = [s[~s.index.duplicated(keep='first')] for s in series]
				all_series_updated.append([s[~s.index.duplicated(keep='first')] for s in series])
				min_length = min(np.min([len(s) for s in updated_series]), min_length)
			min_length = int(min_length)
			all_series = all_series_updated

			for series in all_series:
				trunc_series = [s[:min_length] for s in series]
				all_series_pd.append(pd.concat(trunc_series, axis=1).interpolate(method='linear')*args.scale)

			df = reduce(lambda x, y: x.add(y, fill_value=0), all_series_pd)/len(PROCGEN_ENVS)
			ewm = df.ewm(alpha=args.alpha, ignore_na=True).mean()

			all_x = np.array([i for i in df.index])
			max_x = max(all_x)
			plt_x = all_x
			plt_y_avg = np.array([y for y in ewm.mean(axis=1)])
			plt_y_std = np.array([std for std in ewm.std(axis=1, ddof=1)])

			# Plot
			ax.plot(plt_x, plt_y_avg, linewidth=args.linewidth, label=meta[-1], linestyle=args.linestyle)
			ax.fill_between(plt_x, plt_y_avg - plt_y_std, plt_y_avg + plt_y_std, alpha=0.1)

			# Add info for sigtest
			info = {'max_x': max_x, 'all_x': all_x, 'avg_y': plt_y_avg, 'std_y': plt_y_std, 'df':ewm, 'tag': results_metas[i][-1]}
			infos_dict[args.results_path[i]].append(info)
		else:
			info = plot_results_for_prefix(args, plt, rp, xpid_prefix, l, tag=args.env_name)
			infos_dict[args.results_path[i]].append(info)
			max_x = max(info['max_x'], max_x)

	all_x = info['all_x']

	all_ax = ax if args.grid else [plt]
	for subax in all_ax:
		if args.threshold is not None:
			threshold_x = np.linspace(0, max_x, 2)
			subax.plot(threshold_x, args.threshold*np.ones(threshold_x.shape), 
				zorder=1, color='k', linestyle='dashed', linewidth=args.linewidth, alpha=0.5, label=args.threshold_label)

	if args.grid:
		handles, labels = all_ax[0].get_legend_handles_labels()
		fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5,1), prop={'size': args.fontsize})
		fig.text(0.5, 0.01, args.x_label, ha='center', fontsize=args.fontsize)
		fig.text(0.0, 0.5, args.y_label, va='center', rotation='vertical', fontsize=args.fontsize)
		for ax in all_ax:
			format_subplot(args, max_x, all_x, fig, ax)
	else:
		format_plot(args, max_x, all_x, fig, plt)

	ALL_ENVS = PROCGEN_ENVS if not args.env_name else [args.env_name]
	if len(args.results_path) > 1:
		if not args.avg_procgen:
			sigtest_results_dict = {k:{env:None for env in ALL_ENVS} for k in args.results_path[1:]}
		else:
			sigtest_results_dict = {k:{results_metas[i+1][-1]:None} for i, k in enumerate(args.results_path[1:])}

		baseline_key = args.results_path[0]
		num_rows = infos_dict[baseline_key][-1]['df'].shape[0]
		row_interval = 1
		for key in infos_dict.keys():
			if key == baseline_key: continue

			for i, info in enumerate(infos_dict[key]):
				baseline_df = infos_dict[baseline_key][i]['df']
				baseline_outcomes = baseline_df.iloc[::-1].iloc[::row_interval].values.tolist()

				treatment_df = info['df']

				if key == 'ucb_drac_plr':
					treatment_outcomes = treatment_df.iloc[::-1].iloc[::10].values.tolist()
					baseline_outcomes = baseline_outcomes[:len(treatment_outcomes)]
				else:
					treatment_outcomes = treatment_df.iloc[::-1].iloc[::row_interval].values.tolist()

				# Welch t-test
				sigtest_results = scipy.stats.ttest_ind(baseline_outcomes, treatment_outcomes, equal_var=False, axis=1)
				sigtest_results_dict[key][info['tag']] = sigtest_results

		fields = ['env', 'p-value', 'k-fwer significant', 'reject ratio']
		for key, results in sigtest_results_dict.items():
			print(f'\nCondition: {key}')
			print('-'*84)
			print(f'{fields[0]:12}{fields[1]:24}{fields[2]:24}{fields[3]:24}')
			print('-'*84)
			for i, (env, r) in enumerate(results.items()):
				sigmarker = '' if r.pvalue[0] >= args.pvalue else '*'

				# k-FWER Welch t-test
				num_reject = len([p for p in r.pvalue if p < args.pvalue*args.kfwer_ratio])
				reject_ratio = num_reject/num_rows
				curve_sig = '*' if reject_ratio > args.kfwer_ratio else ' '

				final_pvalue_str = str(r.pvalue[0]) + sigmarker

				if reject_ratio > 1:
					raise ValueError('Reject ratio > 1.0')
				print(f'{env:12}{final_pvalue_str:24}{curve_sig:24}{reject_ratio:24}')

	# Render plot
	if args.savename:
		if not os.path.exists(args.save_path):
			os.makedirs(args.save_path, exist_ok=True)

		plt.savefig(os.path.join(args.save_path, f'{args.savename}.pdf'), bbox_inches="tight", pad_inches=0, dpi=dpi)
	else:
		plt.show()
