"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 2000
LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]  # you can try different target replacement strategies
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32

ENV_NAME = 'Pendulum-v1'


class Actor_Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs, action_bound):
        super(Actor_Net, self).__init__()
        self.action_bound = action_bound
        self.l1 = nn.Linear(n_features, n_hidden)
        self.mu = nn.Linear(n_hidden, n_outputs)

        nn.init.normal(self.l1.weight, 0., 0.3)
        nn.init.constant_(self.l1.bias, 0.3)
        nn.init.normal(self.mu.weight, 0., 0.3)
        nn.init.constant_(self.mu.bias, 0.3)

        self.optimizer = optim.Adam(self.parameters(), lr=LR_A)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        mu = self.mu(x)
        action = torch.tanh(mu)

        scaled_a = torch.multiply(action, self.action_bound)
        return scaled_a


class Critic_Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=LR_C)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        h2 = F.relu(self.fc2(x))
        q = self.fc3(h2)
        return q


class Agent(object):
    def __init__(self, action_dim, action_bound, learning_rate, replacement):
        self.a_dim = action_dim
        action_bound = torch.tensor(action_bound)
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        # input s, output a
        self.a_behavior = Actor_Net(state_dim, 30, self.a_dim, action_bound)

        # input s_, get a_ for critic
        self.a_target = Actor_Net(state_dim, 30, self.a_dim, action_bound)

        self.c = Critic_Net(state_dim+action_dim, 30, 1)

        self.c_target = Critic_Net(state_dim+action_dim, 30, 1)

        self.t_replace_counter = 0

    def learn(self, s, a, r, s_):  # batch update
        s = torch.tensor(s).to(torch.float32)
        a = torch.tensor(a).to(torch.float32)
        r = torch.tensor(r).to(torch.float32)
        s_ = torch.tensor(s_).to(torch.float32)
        new_a = self.a_target(s)
        critic_value_ = self.c_target(s_, new_a)

        critic_value = self.c(s, a)
        td_target = r + GAMMA*critic_value_
        critic_loss = F.mse_loss(td_target, critic_value)
        self.c.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.c.optimizer.step()

        pi = self.a_behavior.forward(s)
        actor_loss = self.c.forward(s, pi).flatten()
        actor_loss = -torch.mean(actor_loss)
        self.a_behavior.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.a_behavior.optimizer.step()

        self.update_target_network()

    def choose_action(self, s):
        return self.a_behavior(torch.tensor(s))

    def update_target_network(self):
        if self.replacement['name'] == 'hard':
            self.t_replace_counter += 1
            self.a_target.load_state_dict(self.a_behavior.state_dict())
        else:
            target_actor_params = self.a_target.named_parameters()
            actor_params = self.a_behavior.named_parameters()

            target_actor_state_dict = dict(target_actor_params)
            actor_state_dict = dict(actor_params)
            for name in actor_state_dict:
                actor_state_dict[name] = self.replacement['tau'] * actor_state_dict[name].clone() + \
                                         (1 - self.replacement['tau']) * target_actor_state_dict[name].clone()
            self.a_target.load_state_dict(actor_state_dict)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter += 1
            self.c_target.load_state_dict(self.c.state_dict())
        else:
            target_critic_params = self.c_target.named_parameters()
            critic_params = self.c.named_parameters()

            target_critic_state_dict = dict(target_critic_params)
            critic_state_dict = dict(critic_params)
            for name in critic_state_dict:
                critic_state_dict[name] = self.replacement['tau'] * critic_state_dict[name].clone() + \
                                         (1 - self.replacement['tau']) * target_critic_state_dict[name].clone()
            self.c_target.load_state_dict(critic_state_dict)

#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


if __name__ == "__main__":
    var = 3  # control exploration
    explore = True
    MODELFILE = "ok.pkl"
    TRAIN = False

    if TRAIN:
        env = gym.make(ENV_NAME)
    else:
        env = gym.make(ENV_NAME, render_mode="human")
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)
    agent = Agent(action_dim, action_bound, LR_A, REPLACEMENT)

    if not TRAIN:
        agent.a_behavior.load_state_dict(torch.load(MODELFILE))
        print(f"loaded from {MODELFILE}")

    t1 = time.time()
    ep_previous = -1500
    for i in range(MAX_EPISODES):
        s, _ = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = agent.choose_action(s)
            if explore and TRAIN:
                a = np.clip(np.random.normal(a.detach().numpy(), var), -2,
                            2)  # add randomness to action selection for exploration
            else:
                a = a.detach().numpy()

            s_, r, done, _, info = env.step(a)
            if TRAIN:
                M.store_transition(s, a, r / 10, s_)

                if M.pointer > MEMORY_CAPACITY:
                    var *= .9995  # decay the action randomness
                    b_M = M.sample(BATCH_SIZE)
                    b_s = b_M[:, :state_dim]
                    b_a = b_M[:, state_dim: state_dim + action_dim]
                    b_r = b_M[:, -state_dim - 1: -state_dim]
                    b_s_ = b_M[:, -state_dim:]

                    agent.learn(b_s, b_a, b_r, b_s_)
                    if j == MAX_EP_STEPS - 1:
                        print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                        if ep_reward > -300 and ep_reward > ep_previous:
                            torch.save(agent.a_behavior.state_dict(), MODELFILE)
                            ep_previous = ep_reward
                        break
            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )

    print('Running time: ', time.time() - t1)
