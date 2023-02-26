python run_experiment.py --mode test --experiment_dir ./runs --experiment_name dubins3dpooravoidtube --dt 0.0025 --checkpoint_toload -1 --num_scenarios 100000 --num_violations 1000 --control_type value --data_step plot_basic_recovery

python run_experiment.py --mode test --experiment_dir ./runs --experiment_name dubins3dpooravoidtube --dt 0.0025 --checkpoint_toload -1 --num_scenarios 100000 --num_violations 1000 --control_type value --data_step plot_binned_recovery

python run_experiment.py --mode test --experiment_dir ./runs --experiment_name dubins3dexpertavoidtube --dt 0.0025 --checkpoint_toload -1 --num_scenarios 100000 --num_violations 1000 --control_type value --data_step plot_basic_recovery

python run_experiment.py --mode test --experiment_dir ./runs --experiment_name dubins3dexpertreachtube --dt 0.0025 --checkpoint_toload -1 --num_scenarios 100000 --num_violations 1000 --control_type value --data_step plot_basic_recovery

python run_experiment.py --mode test --experiment_dir ./runs --experiment_name multivehiclecollisionavoidtube --dt 0.0025 --checkpoint_toload -1 --num_scenarios 100000 --num_violations 1000 --control_type value --data_step plot_basic_recovery

python run_experiment.py --mode test --experiment_dir ./runs --experiment_name rocketlandingreachtube --dt 0.0025 --checkpoint_toload 199000 --num_scenarios 100000 --num_violations 1000 --control_type value --data_step plot_basic_recovery