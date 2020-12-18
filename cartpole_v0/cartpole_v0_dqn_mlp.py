import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('CartPole-v0').unwrapped

# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if(len(self.memory) < self.capacity):
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(inputs, 32)
        self.l2 = nn.Linear(32, 128)
        self.l3 = nn.Linear(128, 32)
        self.lout = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.lout(x)


# Training
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.0
EPS_DECAY = 200
TARGET_UPDATE = 3

n_state = 4  # (x, dx, theta, dtheat)
n_action = 2  # (0, 1)

policy_net = DQN(n_state, n_action).to(device)
target_net = DQN(n_state, n_action).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    # input : 1x4 state
    # output : 1x1 action

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # action from policy
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_action)]], device=device, dtype=torch.long)
        # random action

# Training Loop


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)  # Batchsize x 4
    action_batch = torch.cat(batch.action)  # Batchsize x 1
    reward_batch = torch.cat(batch.reward)  # Batchsize

    # Bellman update
    Q_s_a = policy_net(state_batch).gather(1, action_batch)  # Batchsize x 1

    V_s_prime = torch.zeros(BATCH_SIZE, device=device)
    V_s_prime[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()

    Q_s_a_expected = reward_batch + (GAMMA * V_s_prime)

    # Compute Huber loss
    loss = F.smooth_l1_loss(Q_s_a, Q_s_a_expected.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


########################
# Plot duration
episode_durations = []
episode_durations_mean = [0] * 99


def plot_durations():
    plt.figure(1)
    plt.clf()
    plt.title('Durations')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)

    if(len(episode_durations) >= 100):
        last_durations = np.array(episode_durations[-100:-1])
        episode_durations_mean.append(np.mean(last_durations))
        plt.plot(episode_durations_mean)

    plt.pause(0.001)


########################
# Main loop
num_episodes = 10000

t_start = time.time()
for i_episode in range(num_episodes):
    state = env.reset()

    duration = 0
    state = torch.tensor([state], dtype=torch.float, device=device)
    for t in count():
        action = select_action(state)
        env.render()
        state_next, reward, done, _ = env.step(action.item())
        state_next = torch.tensor(
            [state_next], dtype=torch.float, device=device)
        reward = torch.tensor([reward], device=device)

        # observe new state
        if done:
            state_next = None

        memory.push(state, action, state_next, reward)

        # update state
        state = state_next

        # optimize policy_net
        optimize_model()

        duration += 1
        if done:
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Plot durations
    episode_durations.append(duration)
    plot_durations()

    print("episode : {0}, duration : {1}, calc time: {2:.1f}".format(
        i_episode, duration, time.time() - t_start))
