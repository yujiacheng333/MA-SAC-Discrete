
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


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, device='cpu'):
        super(PolicyNet, self).__init__()
        self.action_dim = action_dim
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.model = torch.nn.Sequential(*[torch.nn.Linear(state_dim, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, action_dim)])

    def forward(self, x, require_log_prob=False):
        temp = self.model(x)
        if require_log_prob:
            return torch.nn.functional.log_softmax(temp, dim=-1)
        else:
            return torch.nn.functional.softmax(temp, dim=-1)

    def action(self, state, deterministically=False):
        prob = self.forward(state).detach().numpy().squeeze()
        if deterministically:
            discrete_action = np.argmax(prob, axis=-1)
        else:
            discrete_action = np.random.choice(np.arange(self.action_dim), p=prob)
        return discrete_action


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
        x = self.model(x)
        return x.reshape([-1, self.action_dim])


class ValueNet(torch.nn.Module):
    """
    V(state), used to cal cr env score
    """
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.model = torch.nn.Sequential(*[torch.nn.Linear(state_dim, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 1)])

    def forward(self, x):
        return torch.squeeze(self.model(x))


class SAC:
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
                 target_update_interval=1,
                 init_entropy_weight=.5,
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
        self.entropy_weight = torch.ones([1]) * init_entropy_weight

        self.step_counter = 0
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape[0]
        self.accumulate_reward_max = - np.inf

        self.q0_net = Qnet(self.obs_dim, self.action_dim)
        self.q1_net = Qnet(self.obs_dim, self.action_dim)

        self.v_net = ValueNet(self.obs_dim)
        self.v_net_tar = deepcopy(self.v_net)
        self.v_net_tar.requires_grad_(False)

        self.policy_net = PolicyNet(self.obs_dim, self.action_dim)

        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q0_optim = torch.optim.Adam(self.q0_net.parameters(), lr=self.lr)
        self.q1_optim = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr)
        self.v_optim = torch.optim.Adam(self.v_net.parameters(), lr=self.lr)
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
        v_target = self.v_net_tar(next_state)
        # update soft-Q net
        q_0_est = self.q0_net(state)
        q_1_est = self.q1_net(state)
        q_target = reward + v_target * self.gamma * done
        pred_q0 = (action_one_hot * q_0_est).sum(-1)
        pred_q1 = (action_one_hot * q_1_est).sum(-1)
        pred_q0_l = tf.smooth_l1_loss(input=pred_q0, target=q_target.detach())
        pred_q1_l = tf.smooth_l1_loss(input=pred_q1, target=q_target.detach())
        self.q0_optim.zero_grad()
        pred_q0_l.backward()
        self.q0_optim.step()
        self.q1_optim.zero_grad()
        pred_q1_l.backward()
        self.q1_optim.step()
        # update V net
        log_prob = self.policy_net(state, require_log_prob=True)
        prob = torch.exp(log_prob)
        combine_q = torch.min(q_0_est, q_1_est)
        v_cr_target = (prob * (combine_q - self.entropy_weight * log_prob)).sum(-1)
        v_cr = self.v_net(state)
        v_loss = tf.smooth_l1_loss(input=v_cr, target=v_cr_target.detach())
        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()
        # update policy
        p_l = - (prob * (combine_q.detach() - self.entropy_weight * log_prob)).sum(-1).mean()
        self.policy_optim.zero_grad()
        p_l.backward()
        self.policy_optim.step()
        self.v_net_tar.load_state_dict(self.polyak_dict(self.v_net, self.v_net_tar, tau=.01))

    def learn(self, step):
        self.step_counter = 0
        while self.step_counter < step + self.learn_starts:
            state = self.env.reset()
            for local_step in range(200):
                action = self.policy_net.action(self._state_reformat(state), deterministically=False)
                next_state, reward, done, log_info = self.env.step(action)
                reward = -2. if done else 0
                if self.accumulate_reward_max < reward:
                    self.accumulate_reward_max = reward
                self.replay_buffer.push(tuple([state, action, reward, next_state, done]))
                state = next_state
                self.step_counter += 1
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
                action = self.policy_net.action(self._state_reformat(state), deterministically=True)
                next_state, reward, done, log_info = self.env.step(action)
                state = next_state
                if done:
                    break
            m_reward.append(i)
        return m_reward


if __name__ == '__main__':

    import gym
    env = gym.make('CartPole-v1')
    model = SAC(env)
    model.learn(1e4)
    print(model.eval(20))
