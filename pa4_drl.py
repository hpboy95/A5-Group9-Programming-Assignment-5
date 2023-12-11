# reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
EPS = 0.9
TAU = 0.005
LR = 1e-4  # Just recall this is the learning rate

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)  # This is 4 values. According to the environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <----- Need to define device
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())  # state_dict saves all parameters
# ^---- This is the list of weights and biases of the different layers

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state: torch.Tensor) -> int:
    """
    Implement Epsilon Greedy

    @param state: The state is a tuple of observation values
    @return: An Action
    """

    # Professor said to use this code
    global steps_done
    sample = random.random()
    steps_done += 1
    if sample > EPS:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    # result = list(policy_net.forward(state).detach())
    # small_probability = EPS / n_actions
    # probability = (1 - EPS) + small_probability
    # prob_list = [small_probability] * n_actions
    # prob_list[result.index(max(result))] = probability
    # action_list = list(range(n_actions))
    # final_result = random.choices(population=action_list, weights=prob_list)[0]
    # global steps_done
    # steps_done += 1
    # return torch.tensor([[final_result]])


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    # print(non_final_mask)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    # print("inside optimize")
    # print(batch.state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # print(next_state_values[non_final_mask])
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


episode_durations = []


def plot(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too (OLD)
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    # Take 100 episode averages and plot them too (New)
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig("ROFL")


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 600
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # TODO: Generate an episode (follow DQN implementation)
    returns = 0
    for t in count():
        # TODO: Select an Action
        new_action = select_action(state)
        # TODO Obtain successive state s, reward
        observation, reward, done, info, _ = env.step(new_action.item())
        returns = reward + (GAMMA * returns)
        reward = torch.tensor([reward], device=device)
        # TODO: Store the transition in memory
        next = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        memory.push(state, new_action, next, reward)

        # TODO: Move to the next state
        state = next

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # TODO: Update the target network every 100 episodes (follow DQN implementation)
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        # Old Code
        # if num_episodes % 100 == 0:
        #     for key in policy_net_state_dict:
        #         target_net_state_dict[key] = policy_net_state_dict[key]
        #     target_net.load_state_dict(target_net_state_dict)

        # New Code
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)


        if done:
            episode_durations.append(t + 1)
            plot()
            break

print('Complete')
