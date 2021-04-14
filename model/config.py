import argparse

parser_do = argparse.ArgumentParser()
parser_do.add_argument('--env_name', default='RLWWTP-v0')
parser_do.add_argument('--tau', default=0.005, type=float)
parser_do.add_argument('--target_update_interval', default=1, type=int)
# parser_do.add_argument('--learning_rate', default=5e-5, type=float)
parser_do.add_argument('--gamma', default=0.99, type=float)
parser_do.add_argument('--capacity', default=1200, type=int)  # replay buffer 100/64
parser_do.add_argument('--batch_size', default=256, type=int)
parser_do.add_argument('--seed', default=False, type=bool)
parser_do.add_argument('--random_seed', default=10, type=int)

parser_do.add_argument('--log_interval', default=50, type=int)
parser_do.add_argument('--load', default=False, type=bool)
# parser_do.add_argument('--exploration_noise', default=[0.5, 5], type=list)  # noise is important
parser_do.add_argument('--max_episode', default=1000, type=int)
parser_do.add_argument('--max_length_of_trajectory', default=10, type=int)  # length of each trajectory
parser_do.add_argument('--print_log', default=1, type=int)
parser_do.add_argument('--update_iteration', default=10, type=int)
