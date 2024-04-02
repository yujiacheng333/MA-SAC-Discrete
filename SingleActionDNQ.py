
import os
import torch.nn.functional as tf
import torch
import collections
import random
import numpy as np
from stable_baselines3.common.utils import get_linear_fn
from copy import deepcopy
from utils import one_hot


class ReplayBeffer:
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return np.asarray(state_list), \
               np.asarray(action_list), \
               np.asarray(reward_list), \
               np.asarray(next_state_list), \
               np.asarray(done_list)

    def buffer_len(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """
    Q(action, state), used to cal env-action pair score
    """

    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.action_dim = action_dim
        self.model = torch.nn.Sequential(*[torch.nn.Linear(state_dim, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, action_dim)])

    def forward(self, x):
        return self.model(x)

    def action(self, x, eps):
        with torch.no_grad():
            if np.random.uniform(0, 1) < eps:
                return np.random.randint(low=0, high=self.action_dim)
            else:
                x = self.model(x).squeeze().detach().cpu().numpy()
                return np.argmax(x, axis=-1)

    def predict(self, x):
        with torch.no_grad():
            x = self.model(x).squeeze().detach().cpu().numpy()
            return np.argmax(x, axis=-1)


class DQN:
    """
        :param env: The environment to learn from (if registered in Gym, can be str)
        :param learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        :param buffer_size: size of the replay buffer
        :param learning_starts: how many steps of the model to collect transitions for before learning starts
        :param batch_size: Minibatch size for each gradient update
        :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
        :param gamma: the discount factor
        :param train_freq: Update the model every ``train_freq`` steps.
        :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        :param target_update_interval: update the target network every ``target_update_interval``
            environment steps.
        :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
        :param exploration_initial_eps: initial value of random action probability
        :param exploration_final_eps: final value of random action probability
        """

    def __init__(self,
                 env,
                 learning_rate=5e-4,
                 buffer_size=5000,
                 learning_starts=1000,
                 batch_size=128,
                 tau=1.0,
                 gamma=0.99,
                 train_freq=4,
                 gradient_steps=1,
                 target_update_interval=100,
                 exploration_fraction=0.1,
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.05,
                 max_grad_norm=10):
        self.env = env
        self.lr = learning_rate
        self.replay_buffer = ReplayBeffer(buffer_maxlen=buffer_size)
        self.learn_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration = exploration_initial_eps
        self.exploration_func = get_linear_fn(exploration_initial_eps,
                                              exploration_final_eps,
                                              exploration_fraction)
        self.step_counter = 0
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.accumulate_reward_max = - np.inf
        self.policy_net = Qnet(self.obs_dim, self.action_dim)
        self.target_net = deepcopy(self.policy_net)
        self.target_net.requires_grad_(False)
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.max_grad_norm = max_grad_norm

    @staticmethod
    def _state_reformat(state, batched=False):
        state = state.astype(np.float32)
        state = state.reshape([1, -1]) if not batched else state.reshape([state.shape[0], -1])
        state = torch.from_numpy(state)
        return state

    @staticmethod
    def _get_done_mask(done):
        done = torch.from_numpy(1 - done.astype(np.float32))
        return done

    @staticmethod
    def polyak_dict(net, target_net, tau):
        with torch.no_grad():
            org_state_dict = net.state_dict()
            tar_state_dict = target_net.state_dict()
            for k in org_state_dict:
                tar_state_dict[k] = org_state_dict[k] * tau + tar_state_dict[k] * (1 - tau)
        return tar_state_dict

    def model_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = self._state_reformat(state, batched=True)
        next_state = self._state_reformat(next_state, batched=True)
        action_one_hot = torch.from_numpy(one_hot(action, depth=self.action_dim))
        reward = torch.from_numpy(reward.astype(np.float32))
        done = self._get_done_mask(done)
        target_v, _ = torch.max(self.target_net(next_state), dim=-1)
        target_q = reward + self.gamma * done * target_v
        est_q = self.policy_net(state)
        est_q = torch.sum(est_q * action_one_hot, dim=-1)
        loss = tf.smooth_l1_loss(est_q, target_q.detach())
        self.policy_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optim.step()
        if self.step_counter % self.target_update_interval == 0:
            target_state_dict = self.polyak_dict(net=self.policy_net, target_net=self.target_net, tau=1.)
            self.target_net.load_state_dict(target_state_dict)

    def learn(self, step):
        self.step_counter = 0
        while self.step_counter < step + self.learn_starts:
            state = self.env.reset()
            for local_step in range(200):
                action = self.policy_net.action(self._state_reformat(state), eps=self.exploration)
                next_state, reward, done, log_info = self.env.step(action)
                reward = -2. if done else 0
                if self.accumulate_reward_max < reward:
                    self.accumulate_reward_max = reward
                self.replay_buffer.push(tuple([state, action, reward, next_state, done]))
                state = next_state
                self.step_counter += 1
                self.exploration = self.exploration_func(1 - self.step_counter/(step + self.learn_starts))
                if self.step_counter > self.learn_starts and self.step_counter % self.train_freq == 0:
                    for _ in range(self.gradient_steps):
                        self.model_update()
                if done:
                    break

    def eval(self, epoch):
        m_reward = []
        for _ in range(epoch):
            state = self.env.reset()
            for i in range(200):
                action = self.policy_net.predict(self._state_reformat(state))
                next_state, reward, done, log_info = self.env.step(action)
                state = next_state
                if done:
                    break
            m_reward.append(i)
        return m_reward

    def save(self):
        os.makedirs('./ckpt', exist_ok=True)
        pack = {'q_net': self.policy_net.state_dict(),
                'q_net_tar': self.target_net.state_dict(),
                'q_optim': self.policy_optim.state_dict()}
        torch.save(pack, './ckpt/model.pth.tar')

    def load(self):
        pack = torch.load('./ckpt/model.pth.tar')
        self.policy_net.load_state_dict(pack['q_net'])
        self.target_net.load_state_dict(pack['q_net_tar'])
        self.policy_optim.load_state_dict(pack['q_optim'])


if __name__ == '__main__':

    import gym
    env = gym.make('CartPole-v1')
    model = DQN(env)
    model.learn(1e4)
    print(model.eval(20))
