# Prioritized Level Replay

### Requirements
```
conda create -n level-replay python=3.8
conda activate level-replay

pip install -r requirements.txt

git clone https://github.com/anonymouscollective/baselines.git
cd baselines 
python setup.py install
cd ..

git clone https://github.com/anonymouscollective/procgen.git
cd procgen 
python setup.py install
cd ..

git clone https://github.com/anonymouscollective/gym-minigrid.git
cd gym-minigrid
pip install -e .
cd ..
```

Note that you may run into cmake finding an incompatible version of g++. You can manually specify the path to a compatible g++ by setting the path to the right compiler in `procgen/procgen/CMakeLists.txt` before the line `project(codegen)`:
```
...
# Manually set the c++ compiler here
set(CMAKE_CXX_COMPILER "/share/apps/gcc-9.2.0/bin/g++")

project(codegen)
...
```

### Training command syntax
| Command-line argument| Description |
| ----------- | ----------- |
|xpid|Name of experiment, used for naming results folder containing experiment output files|
|env_name|Name of environment (lowercase Procgen game names or one of "MultiRoom-N4-S7-Random", "ObstructedMazeGamut-Easy-v0", "ObstructedMazeGamut-Medium-v0")|
|distribution_mode|Difficulty setting for Procgen. Use "easy" or "hard".|
|use_gae|Uses GAE if passed|
|gamma|MDP discount factor|
|num_train_seeds|Number of training levels|
|num_test_seeds|Number of test levels|
|seed|Experiment PRNG seed|
|arch|Either "large" or "small". Experiments in our paper were ran using "large".|
|lr|PPO learning rate|
|num_env_steps|Total number of training steps (summed across all actors)|
|num_steps|PPO rollout length|
|num_processes|Number of PPO actors|
|ppo_epoch|Number of PPO epochs per update cycle|
|num_mini_batch|Number of mini-batches per PPO epoch|
|level_replay_strategy|Scoring function for level replay. Set to "random" for uniform sampling. Otherwise, can use one of the following scoring functions: "policy entropy", "least confidence", "min_margin", "gae", "value_l1", "one_step_td_error." Can also set to "tscl_window" to train using the TSCL Window algorithm.|
|level_replay_score_transform|Set to "rank" or "power" for rank or proportional prioritization respectively. Can also set to "eps_greedy" when using epsilon-greedy TSCL Window.|
|level_replay_temperature|Beta (temperature) parameter for level replay|
|staleness_coef|Staleness coefficient for level replay|
|level_replay_eps|Set this to the exploration probability for epsilon-greedy TSCL Window|
|train_full_distribution|Samples from full level distribution at training if passed|
|level_replay_seed_buffer_size|Number of top levels in terms of learning potential to track if training on the full level distribution|
|level_replay_prob|Probability of sampling from replay distribution when training on the full level distribution|
|use_ucb|Uses UCB-DrAC if passed|
|use_mixreg|Uses mixreg if passed|
|ucb_window_length|Window size for UCB-DrAC|
|ucb_exploration_coef|Exploration coefficient for UCB-DrAC|
|tscl_window_size|Window size for TSCL Window|
|verbose|Prints out additional information to stdout during training if passed|


### Examples
##### Train on a Procgen environment, interpolating between value_l1 (primary) and uncertainty (secondary, 20% of the time)
```
python -m train \
--xpid=test_strategy_interpolation \
--env_name=bigfish \
--distribution_mode=easy \
--use_gae=True \
--gamma=0.999 \
--num_train_seeds=200 \
--num_test_seeds=10 \
--final_num_test_seeds=100 \
--seed=5 \
--train_full_distribution=False \
--level_replay_seed_buffer_size=0 \
--level_replay_seed_buffer_priority=score \
--arch=small \
--lr=0.0005 \
--num_env_steps=25000000 \
--num_steps=256 \
--num_processes=64 \
--ppo_epoch=3 \
--num_mini_batch=8 \
--algo=ppo \
--level_replay_strategy=value_l1 \
--level_replay_max_score_coef=0.0 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--level_replay_schedule=proportionate \
--level_replay_rho=1.0 \
--level_replay_prob=0.95 \
--level_replay_alpha=1.0 \
--level_replay_secondary_strategy uncertainty \
--level_replay_strategy_mix_coef 0.2 \
--level_replay_secondary_score_transform eps_greedy \
--level_replay_secondary_temperature 1.0 \
--level_replay_secondary_eps 0.0 \
--secondary_staleness_coef 0.0 \
--staleness_transform=power \
--staleness_temperature=1.0 \
--staleness_coef=0.1 \
--log_interval=1 \
--weight_log_interval=10 \
--log_grad_norm=False \
--use_ucb=False \
--use_mixreg=False \
--verbose
``` 

##### Train on a Procgen environment using PPO with uniform sampling
```
python -m train \
--xpid=lr-ppo-track_grad-bigfish-random-s200_0 \
--env_name=starpilot \
--distribution_mode=easy \
--use_gae=True \
--gamma=0.999 \
--num_train_seeds=200 \
--num_test_seeds=10 \
--final_num_test_seeds=100 \
--seed=5 \
--train_full_distribution=False \
--level_replay_seed_buffer_size=0 \
--level_replay_seed_buffer_priority=score \
--arch=large \
--lr=0.0005 \
--num_env_steps=25000000 \
--num_steps=256 \
--num_processes=64 \
--ppo_epoch=3 \
--num_mini_batch=8 \
--algo=ppo \
--level_replay_strategy=random \
--log_interval=10 \
--weight_log_interval=10 \
--log_grad_norm=True \
--use_ucb=False \
--use_mixreg=False \
--verbose
```

##### Train on a Procgen environment using PLR
```
python -m train \
--xpid=lr-ppo-track_grad-bigfish-plr-s200_0 \
--env_name=bigfish \
--distribution_mode=easy \
--use_gae=True \
--gamma=0.999 \
--num_train_seeds=200 \
--num_test_seeds=10 \
--final_num_test_seeds=100 \
--seed=5 \
--train_full_distribution=False \
--level_replay_seed_buffer_size=0 \
--level_replay_seed_buffer_priority=score \
--arch=small \
--lr=0.0005 \
--num_env_steps=25000000 \
--num_steps=256 \
--num_processes=64 \
--ppo_epoch=3 \
--num_mini_batch=8 \
--algo=ppo \
--level_replay_strategy=value_l1 \
--level_replay_max_score_coef=0.0 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--level_replay_schedule=proportionate \
--level_replay_rho=1.0 \
--level_replay_prob=0.95 \
--level_replay_alpha=1.0 \
--level_replay_eps=0.05 \
--staleness_transform=power \
--staleness_temperature=1.0 \
--staleness_coef=0.1 \
--log_interval=1 \
--weight_log_interval=10 \
--log_grad_norm=True \
--use_ucb=False \
--use_mixreg=False \
--verbose
```
##### Train using PLR on the full level distribution of MultiRoom-N4-Random
```
python -m train \
--xpid=lr-ppo-minigrid-multiroom-n4-s7-random-v0-plr-full_dist \
--env_name=MiniGrid-MultiRoom-N4-S7-Random-v0 \
--distribution_mode=easy \
--use_gae=True \
--gamma=0.999 \
--num_train_seeds=200 \
--num_test_seeds=10 \
--final_num_test_seeds=100 \
--seed=5 \
--train_full_distribution=True \
--level_replay_seed_buffer_size=4000 \
--level_replay_seed_buffer_priority=replay_support \
--arch=small \
--lr=0.0007 \
--num_env_steps=100000000 \
--num_steps=256 \
--num_processes=64 \
--ppo_epoch=4 \
--num_mini_batch=8 \
--algo=ppo \
--level_replay_strategy=value_l1 \
--level_replay_max_score_coef=0.0 \
--level_replay_score_transform=rank \
--level_replay_temperature=0.1 \
--level_replay_schedule=fixed \
--level_replay_rho=1.0 \
--level_replay_prob=0.95 \
--level_replay_alpha=1.0 \
--level_replay_eps=0.05 \
--staleness_transform=power \
--staleness_temperature=1.0 \
--staleness_coef=0.3 \
--log_interval=10 \
--weight_log_interval=10 \
--log_grad_norm=False \
--use_ucb=False \
--use_mixreg=False \
--no_ret_normalization=True \
--verbose
```

### Plotting results
You can plot results using `plot.py`. This script also computes the statistical significance between the plotted conditions and the base condition (first condition passed in). The main arguments behave as follows:
| Command-line argument| Description |
| ----------- | ----------- |
| -b | Path to directory containing results folders |
|-r| A list of results folder names, each folder corresponding to one experimental condition, e.g. PPO + PLR with specific hyperparameters, containing all experiment results folders for each training run per environment under that condition, e.g. 5 runs per Procgen game. Experiment results are assumed to follow the naming convention "<xpid_prefix>-<env_name>-*-_[0-9]+", where the final integer value after '_' indicates the run index.|
|-l| A list of labels, one for each experimental setting (same number as list passed to argument -r)|
|--xpid_prefix| A list of <xpid_prefix> values corresponding to the result folder matching pattern in the description for the argument -r|
|--avg_procgen|If -r is a list of results folders containing Procgen results, then normalize and average the y-axis values across runs according to Cobbe et al, 2019.|
|--gap|Plot the difference between train and test return (i.e. the generalization gap) on the y-axis|
|--grid|If -r is a list of results folders containing Procgen results, plot all Procgen results as a grid plot, with a subplot per environment|
|-xi|x-axis tick increment|
|xts|Suffix added to each x-axis tick label|
|-a|The smoothing coefficient for the exponential weighted moving average of y-values|
|-xl|x-axis label|
|-yl|y-axis label|
|--pvalue|Value of p, under which results are deemed statistically significant w.r.t. the base (first) condition.|


#### Plotting examples for Procgen results
##### Grid of test episode return vs training steps per game
```
python plot.py \
-b '~/logs' \
-r random plr \
-l 'Random' 'PLR' \
--xpid_prefix 'lr-ppo-random' 'lr-ppo-plr' \
--grid
-xi 5000000 -xts M --save_width 330 --save_height 330 \
-a 0.1 --fontsize 8 -y 'test:mean_episode_return' -yl 'Mean test episode return'
```

##### Mean normalized train episode return vs training steps
```
python plot.py \
-b '~/logs' \
-r random plr \
-l 'Random' 'PLR' \
--xpid_prefix 'lr-ppo-random' 'lr-ppo-plr' \
--avg_procgen \
-xi 5000000 -xts M --save_width 330 --save_height 330 \
-a 0.1 --fontsize 8  -y 'train_eval:mean_episode_return' -yl 'Mean train episode return'
```

##### Mean normalized test episode return vs training steps
```
python plot.py \
-b '~/logs' \
-r random plr \
-l 'Random' 'PLR' \
--xpid_prefix 'lr-ppo-random' 'lr-ppo-plr' \
--avg_procgen \
-xi 5000000 -xts M --save_width 330 --save_height 330 \
-a 0.1 --fontsize 8  -y 'test:mean_episode_return' -yl 'Mean train episode return'
```

##### Mean generalization gap vs training steps
```
python plot.py \
-b '~/logs' \
-r random plr \
-l 'Random' 'PLR' \
--xpid_prefix 'lr-ppo-random-hard' 'lr-ppo-plr-hard' \
--avg_procgen --gap \
-xi 5000000 -xts M --save_width 330 --save_height 330 \
-a 0.1 --fontsize 8 -yl 'Mean test episode return'
```

#### Plotting examples for MiniGrid results
```
python plot.py \
-b '~/logs' \
-r test_minigrid_random test_minigrid_plr \
--xpid_prefix lr-ppo-small-nonorm \
-l Random PLR \
-xts M \
--save_width 330 --save_height 205 \
--y_label 'Average test episode return' --env_name MultiRoom-N4-Random-v0 -a 0.1
```

##### Stacked area chart of probability mass over difficulties during training on MiniGrid environments
Note that --xpid is the name of the experiment result folder whose level_weights.csv file will be used, and
--resultdir is the directory containing this result folder.
```
python analyze_curriculum.py \
--resultdir '~/logs' \
--env_name "MiniGrid-MultiRoom-N4-S7-Random-v0" \
--xpid "lr-ppo-MiniGrid-MultiRoom-N4-S7-Random-v0-plr-s4000_0" \
--barchart \
--xtick 1 \
--xinc 1 \
```

### Evaluating trained models
##### Evaluate all saved models for Procgen matching experiment id pattern <xpid_template>:
Note that here, the <xpid_template> is used to match experiment results folders (named by experiment ids) and should include the substring "ENV-NAME" that acts as a placeholder for the lowercase Procgen environment names, e.g. "lr-ppo-random-ENV_NAME-s200" would experiment ids like "lr-ppo-random-bigfish-s200_1", "lr-ppo-random-bigfish-s200_2", etc. This script assumes <xpid_template> matches final models trained via the same method and evaluates each saved model matching the template on 100 test episodes, aggregating episodic returns across models by Procgen game. The results are saved as a .csv into `results/<xpid_template>.csv`.
```
python evaluate_models.py \
--verbose \
--env_type procgen \
-b ~/logs/ppo/random \
--xpid_template <xpid_template>
```

##### Compute returns normalized relative to PPO with uniform sampling:
This command computes the normalized test returns relative to the uniform sampling baseline, reported in Tables 2 and 4. Generally it takes a list of .csv files corresponding to evaluation results generated by `evaluate_models.py` and generates a table of mean normalized returns per game and averaged over all games, normalized relative to the first results file. The main arguments work as follows:
| Command-line argument| Description |
| ----------- | ----------- |
| -b | Path to directory containing .csv results files |
|-f | List of evaluation result files generated using `evaluate_models.py`|
|-l | List of method names corresponding to the evaluation results|
|--compare_max|Bold figures that are not statistically significantly different from the max return for each game and overall average return, if passed|
|--mode|Either "latex" or "markdown". The final table output will be in the specified language.|
```
python norm_returns.py \
--sigfigs 1 \
-b results/ \
-f \
lr-ppo-ENV_NAME-random-level_sampler-large-25m-s200.csv \
lr-ppo-plr-ENV_NAME-value_l1-rank-tau0.1-staleness0.1_power_temp1.0-s200.csv \
-l \
Uniform \
PLR \
--compare_max \
--mode latex
```