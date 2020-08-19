import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='RLWWTP-v0')
parser.add_argument('--tau', default=0.005, type=float)
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--capacity', default=100, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=10, type=int)

parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--load', default=False, type=bool)
parser.add_argument('--max_episode', default=500, type=int)
parser.add_argument('--max_length_of_trajectory', default=50, type=int)  # length of each trajectory
parser.add_argument('--print_log', default=1, type=int)
parser.add_argument('--update_iteration', default=10, type=int)

