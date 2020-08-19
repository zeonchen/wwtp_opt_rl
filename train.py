from model.config import parser
import os
import gym
import numpy as np
import torch
from utils.logger import save_output, concatenate_data
from utils.plot import action_plot, reward_plot
from maddpg.network import MADDPG

print(torch.__version__)
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
action_dim = 1
max_action = env.action_space.high
min_action = env.action_space.low
min_val = torch.tensor(1e-7).float().to(device)


def main(i, transfer=False):
    episode_reward = []
    episode_aer = []
    episode_slu = []
    lr = args.learning_rate
    agent_do = MADDPG(state_dim, action_dim, np.array([1.0]), np.array([0.]), phase=2,
                      actor_path='./outputs/ddpg/model/actor_do.pkl',
                      critic_path='./outputs/ddpg/model/critic_do.pkl',
                      actor_lr=lr, critic_lr=lr)  # initialize ddpg
    agent_dosage = MADDPG(state_dim, action_dim, np.array([1.0]), np.array([0.0]), phase=2,
                          actor_path='./outputs/ddpg/model/actor_dosage.pkl',
                          critic_path='./outputs/ddpg/model/critic_dosage.pkl',
                          actor_lr=lr, critic_lr=lr)  # initialize ddpg
    ep_r = 0
    ep_a = 0
    ep_s = 0

    if transfer:
        for name, param in agent_do.actor.named_parameters():
            if name.startswith('l4') or name.startswith('l3') or name.startswith('l2'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in agent_do.critic.named_parameters():
            if name.startswith('l4') or name.startswith('l3') or name.startswith('l2'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in agent_dosage.actor.named_parameters():
            if name.startswith('l4') or name.startswith('l3') or name.startswith('l2'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in agent_dosage.critic.named_parameters():
            if name.startswith('l4') or name.startswith('l3') or name.startswith('l2'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    for epoch in range(args.max_episode):
        state_do = env.reset()
        state_dosage = env.reset()

        t = 0
        while True:
            # parallel 1
            action_do_1 = agent_do.selection_action(state_do)  # generate actions
            action_dosage_1 = agent_dosage.selection_action(state_dosage)

            if transfer:
                noise = max(0.2 - 0.01 * epoch, 0.0)
            else:
                noise = max(0.2 - 0.0004 * epoch, 0.0)

            action_do_1 = (action_do_1 + np.random.normal(0, noise, size=1)).clip(0.0, 1.0) # add noise for exploration
            action_dosage_1 = (action_dosage_1 + np.random.normal(0, noise, size=1)).clip(0.0, 1.0)

            # transition
            next_state_1, reward_1, done_1, info_1 = env.step([action_do_1, action_dosage_1])
            next_state_do = action_do_1
            next_state_dosage = action_dosage_1

            ep_r += reward_1
            ep_a += action_do_1[0]
            ep_s += (action_dosage_1[0])

            agent_do.replay_buffer.push(
                (state_do, next_state_do, [action_do_1, action_dosage_1], reward_1, np.float(done_1)))
            agent_dosage.replay_buffer.push(
                (state_dosage, next_state_dosage, [action_do_1, action_dosage_1], reward_1, np.float(done_1)))

            state_do = next_state_do
            state_dosage = next_state_dosage

            t += 1

            if t >= args.max_length_of_trajectory:
                if epoch % args.print_log == 0:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(epoch, ep_r / t, t))
                episode_reward.append(ep_r / t)
                episode_aer.append(ep_a / t)
                episode_slu.append(ep_s / t)
                ep_r = 0
                ep_a = 0
                ep_s = 0
                break

        # update network
        if len(agent_do.replay_buffer.storage) >= args.capacity - 1:
            agent_do.update([agent_do, agent_dosage])
            agent_dosage.update([agent_do, agent_dosage])
            # torch.save(agent_do.actor, './outputs/ddpg/model/actor_do.pkl')
            # torch.save(agent_dosage.actor, './outputs/ddpg/model/actor_dosage.pkl')
            # torch.save(agent_do.critic, './outputs/ddpg/model/critic_do.pkl')
            # torch.save(agent_dosage.critic, './outputs/ddpg/model/critic_dosage.pkl')
            save_output(episode_reward=episode_reward, save_path='./outputs/ddpg/reward/ddpg_{}_{}.txt'.format(i, phase))
            save_output(episode_reward=episode_aer, save_path='./outputs/ddpg/aer/aer_{}_{}.txt'.format(i, phase))
            save_output(episode_reward=episode_slu, save_path='./outputs/ddpg/slu/slu_{}_{}.txt'.format(i, phase))


if __name__ == '__main__':
    num = 5
    phase = 1
    for i in range(num):
        print('Recursion {}'.format(i))
        main(i)

    df_reward = concatenate_data('./outputs/ddpg/reward/', num=num, algorithm='DDPG')
    df_aer = concatenate_data('./outputs/ddpg/aer/', num=num, algorithm='aer')
    df_slu = concatenate_data('./outputs/ddpg/slu/', num=num, algorithm='slu')

    df_aer['slu'] = df_slu['slu']

    action_plot(data=df_aer, values=['slu', 'aer'], legend=True, ylabel='Value')
    reward_plot(data=df_reward, values=['DDPG'], ylabel='Reward')
