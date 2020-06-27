from config.ddpg_config import parser
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import save_output, concatenate_data
from utils.plot import action_plot, reward_plot
from itertools import count
from model.rl.ddpg import DDPG
import pandas as pd


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

env = gym.make(args.env_name).unwrapped

# random seed generation
if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

# basic info
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high
min_action = env.action_space.low
min_val = torch.tensor(1e-7).float().to(device)


class Surrogate(nn.Module):
    def __init__(self):
        super(Surrogate, self).__init__()
        # fully connected part: network structure
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 16)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.bn1(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = F.relu(self.bn2(self.fc4(x)))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)

        return x


def main(i, phase, transfer=False):
    episode_reward = []
    episode_aer = []
    episode_slu = []
    agent = DDPG(state_dim, action_dim, max_action, min_action, phase=phase)  # initialize ddpg
    ep_r = 0
    ep_a = 0
    ep_s = 0
    max_r = -10
    df = pd.DataFrame([0] * 10).T

    if transfer:
        for name, param in agent.actor.named_parameters():
            if name.startswith('l4'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in agent.critic.named_parameters():
            if name.startswith('l4'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    print("====================================")
    print("Collection Experience...")
    print("====================================")

    state = env.reset()
    max_reward = -1.0
    for epoch in range(args.max_episode):
        # pre-sampling, acquire mu and std
        for t in count():
            action = agent.selection_action(state)  # generate actions
            action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high
            )  # add noise for exploration

            next_state, reward, done, info = env.step(action)

            # Q = agent.critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device),
            #                       torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device))
            # if max_r < reward:
            #     max_r = reward
            #     print(max_r, action)

            # output = np.concatenate([action, [Q.item()], [reward]])
            # output = pd.DataFrame(output).T
            # df = df.append(output)
            if reward > max_reward:
                max_reward = reward
                print(action, reward)
            ep_r += reward
            ep_a += action[0]
            ep_s += action[1]
            agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

            state = next_state

            if t >= args.max_length_of_trajectory:
                # df.to_excel('D:\\rl_wwtp\\outputs\\tsne.xlsx')
                if epoch % args.print_log == 0:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}, \t{}".format(epoch, ep_r / t, t, max_r))
                episode_reward.append(ep_r / t)
                episode_aer.append(ep_a / t)
                episode_slu.append(ep_s / t)
                ep_r = 0
                ep_a = 0
                ep_s = 0
                break

        # update network
        if len(agent.replay_buffer.storage) >= args.capacity - 1:
            if phase == 1:
                agent.update()
            torch.save(agent.actor, './outputs/ddpg/model/actor.pkl')
            torch.save(agent.critic, './outputs/ddpg/model/critic.pkl')
            # save_output(episode_reward=episode_reward, save_path='./outputs/ddpg/res/reward/ddpg_{}_{}.txt'.format(i, phase))
            # save_output(episode_reward=episode_aer, save_path='./outputs/ddpg/res/aer/aer_{}_{}.txt'.format(i, phase))
            # save_output(episode_reward=episode_slu, save_path='./outputs/ddpg/res/slu/slu_{}_{}.txt'.format(i, phase))


if __name__ == '__main__':
    num = 3
    phase = 1
    for i in range(num):
        print('Recursion {}'.format(i))
        # if phase == 2:
        #     args.exploration_noise = [0.1, 1]
        main(i, phase=phase)

    df_reward = concatenate_data('./outputs/ddpg/res/reward/', num=num, algorithm='DDPG')
    df_aer = concatenate_data('./outputs/ddpg/res/aer/', num=num, algorithm='aer')
    df_slu = concatenate_data('./outputs/ddpg/res/slu/', num=num, algorithm='slu')

    df_aer['slu'] = df_slu['slu']

    action_plot(data=df_aer, values=['slu', 'aer'], legend=True, ylabel='Value')
    # action_plot(data=df_aer, values=['aer'], legend=True, ylabel='Value')
    reward_plot(data=df_reward, values=['DDPG'], ylabel='Reward')
