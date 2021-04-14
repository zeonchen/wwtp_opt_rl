import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from maddpg.config_do import parser_do

args = parser_do.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()


class ReplayBuffer:
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:  # if buffer is full, cover old data.
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        state_buffer, next_state_buffer, action_buffer, reward_buffer, done_buffer = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            state_buffer.append(np.array(state, copy=False))
            next_state_buffer.append(np.array(next_state, copy=False))
            action_buffer.append(np.array(action, copy=False))
            reward_buffer.append(np.array(reward, copy=False))
            done_buffer.append(np.array(done, copy=False))

        return np.array(state_buffer), np.array(next_state_buffer), np.array(action_buffer), \
               np.array(reward_buffer).reshape(-1, 1), np.array(done_buffer).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, action_dim)

        self.max_action = torch.from_numpy(max_action).to(device)
        self.min_action = torch.from_numpy(min_action).to(device)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = self.max_action * F.sigmoid(self.l4(x))

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 1)

    def forward(self, x, u):
        # a = torch.cat([x, u], 0)
        a = torch.cat([x, u], 1).unsqueeze(0)
        x = torch.relu(self.l1(a))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        # represent = x[0].detach().cpu().numpy()
        x = self.l4(x)
        return x


class MADDPG(object):
    def __init__(self, state_dim, action_dim, max_action, min_action, actor_path='', critic_path='',
                 phase=1, actor_lr=1e-3, critic_lr=1e-3):
        if phase != 1:
            self.actor = torch.load(actor_path, map_location='cuda')
            self.actor.to(device)
        elif phase == 1:
            self.actor = Actor(state_dim, action_dim, max_action, min_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, min_action).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr)

        if phase != 1:
            self.critic = torch.load(critic_path, map_location='cuda')
            self.critic.to(device)
        elif phase == 1:
            self.critic = Critic(state_dim, 2).to(device)
        self.critic_target = Critic(state_dim, 2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_lr)

        self.replay_buffer = ReplayBuffer()
        self.writer = SummaryWriter('.\\outputs\\ddpg\\log\\')
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        # self.num_training = 0

    def selection_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().detach().numpy()[0]

    def update(self, agents):
        for it in range(args.update_iteration):
            state, next_state, action, reward, done = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).squeeze(2).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            do = agents[0].actor_target(next_state)
            dosage = agents[1].actor_target(next_state)
            target_action = torch.cat([do, dosage], 1).float()

            target_Q = self.critic_target(next_state, target_action)
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            current_Q = self.critic(state, action)

            # update critic network
            critic_loss = F.mse_loss(current_Q, target_Q)
            # self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # update actor network
            do = agents[0].actor(state)
            dosage = agents[1].actor(state)
            cur_action = torch.cat([do, dosage], 1).float()
            actor_loss = -self.critic(state, cur_action).mean()
            # self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target network
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
            # self.writer.close()
