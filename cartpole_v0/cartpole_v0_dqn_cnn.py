import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
plt.ion()

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
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#####################################
# Network


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        # print(x.size())  # 1x3x40x90
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.size())  # 1x16x18x43
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.size())  # 1x32x7x20
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size())  # 1x32x2x8
        # print(x.view(x.size(0), -1).size())  # 1x512
        return self.head(x.view(x.size(0), -1))

#################################
# Input extraction


# resize = T.Compose([T.ToPILImage(),
#                     T.Resize(40, interpolation=Image.CUBIC),
#                     T.ToTensor()])

resize = T.Compose([T.ToPILImage(), T.ToTensor()])


def get_cart_location(screen_width):
    # env.state = [x, x_dot, theta, theta_dot] numpy.ndarray (4,)
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    # Returned screen requested by gym is 400x600x3 (HWC), but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # it is return by numpy.ndarray type
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # 3x400x600

    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # 3x160x600
    view_width = int(screen_width * 0.6)  # 360
    cart_location = get_cart_location(screen_width)  # in pixel

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
        # start:None, end:360, step:None
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
        # start:-360, end:None, step:None
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]  # 3x160x360
    # img = Image.fromarray(screen.transpose((1, 2, 0)), 'RGB')
    # img.show()

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)  # torch.Tensor 3x160x360

    # Resize, and add a batch dimension (BCHW) 1x3x40x90
    return resize(screen).unsqueeze(0).to(device)


# env.reset()
# plt.figure(1)
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

#################################
# Training

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.0
EPS_DECAY = 200
TARGET_UPDATE = 4

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

#####################################
# Training Loop


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # [Transition(s1,a1,s1',r1), ... , Transition(sn,an,sn',rn)]
    batch = Transition(*zip(*transitions))
    # *transitions = Transition(s1,a1,s1',r1) ... Transition(sn,an,sn',rn)
    # *zip(*transitions) = (s1,....,sn) (a1,...,an) (s1',...,sn') (r1,...,rn)
    # batch = Transition((s1,...,sn), (a1,...,an), (s1',...,sn'), (r1,...,rn))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    # torch.tensor([s1', s2', ..., sn'])

    # single tensor <- tuple of tensors
    # ex) torch.tensor([1, 2]) <- (torch.tensor([1]), torch.tensor([2]))
    state_batch = torch.cat(batch.state)  # BatchSize x 3 x 40 x 90
    action_batch = torch.cat(batch.action)  # BatchSize x 1
    reward_batch = torch.cat(batch.reward)  # BatchSize

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print(policy_net(state_batch).size())  # BatchSize x 2
    # print(policy_net(state_batch).size()).gather(1, action_batch) # BatchSize x 1

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # target_net(non_final_next_states).max(1)[0] # size of non_final_mask <= BatchSize
    # detach() 를 해야 밑에 loss 계산 할 때 gradient 해서 update를 하지 않음

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

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


#########################
# Main loop
num_episodes = 100000

t_start = time.time()
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    duration = 0
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # state : 1x3x40x90
        # action : 1x1
        # next_state : 1x3x40x90
        # reward : 1

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
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
