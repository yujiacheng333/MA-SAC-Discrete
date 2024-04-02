from torch.distributions import Normal
import torch.nn.functional as tf
import torch
import collections
import random
import numpy as np
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
    def __init__(self, state_dim,
                 action_dim,
                 action_scale=1.,
                 action_bias=0, log_std_min=-20, log_std_max=2, eps=1e-6, device='cpu'):
        """Here the action dim is used as preform diag-GMM model,
        for each step, the logprob can be cal by p(y)dy=p(x)dx, and x=f^{-1}(y)
        to this end p(y)=p(x)tanh^{-1}(y)
        """
        super(PolicyNet, self).__init__()
        self.action_dim = action_dim
        self.eps = eps
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.model = torch.nn.Sequential(*[torch.nn.Linear(state_dim, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 256),
                                           torch.nn.ReLU()])
        self.mu = torch.nn.Linear(256, action_dim)
        self.log_sigma = torch.nn.Linear(256, action_dim)

    def forward(self, x):
        temp = self.model(x)
        mu = self.mu(temp)
        log_sigma = self.log_sigma(temp)
        log_sigma = torch.clip(log_sigma, min=self.log_std_min, max=self.log_std_max)
        distribute = Normal(mu, log_sigma.exp())
        return distribute

    def action(self, state, deterministically=False):
        if deterministically:
            distribute = self(state)
            return distribute.loc, None
        else:
            distribute = self(state)
            x = distribute.rsample()
            y = torch.tanh(x)
            action = y * self.action_scale + self.action_bias
            log_prob = distribute.log_prob(x) - torch.log(self.action_scale * (1 - y.pow(2)) + self.eps)
            log_prob = torch.sum(log_prob, dim=-1)
            return action, log_prob


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
    def __init__(self,
                 env,
                 learning_rate=5e-4,
                 buffer_size=10000,
                 learning_starts=1000,
                 batch_size=128,
                 tau=.01,
                 gamma=0.99,
                 train_freq=4,
                 gradient_steps=1,
                 target_update_interval=1,
                 init_entropy_weight=.5,
                 max_grad_norm=10):
        self.env = env
        self.action_dim = self.env.action_space.shape[-1]
        self.obs_dim = np.prod(self.env.observation_space.sample.shape)

        self.lr = learning_rate
        self.replay_buffer = ReplayBeffer(buffer_maxlen=buffer_size)
        self.learn_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.step_counter = 0
        self.target_update_interval = target_update_interval

        self.entropy_weight = torch.ones(1) * init_entropy_weight
        self.entropy_weight.requires_grad = True
        self.entropy_target = - float(self.action_dim)
        self.ent_optim = torch.optim.Adam([self.entropy_weight], lr=self.lr)

        self.accumulate_reward_max = - np.inf
        self.q0_net = ValueNet(self.obs_dim+self.action_dim)
        self.q1_net = ValueNet(self.obs_dim+self.action_dim)
        self.q0_optim = torch.optim.Adam(self.q0_net.parameters(), lr=self.lr)
        self.q1_optim = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr)

        self.v_net = ValueNet(self.obs_dim)
        self.v_net_tar = deepcopy(self.v_net)
        self.v_net_tar.requires_grad_(False)
        self.v_optim = torch.optim.Adam(self.v_net.parameters(), lr=self.lr)

        self.policy_net = PolicyNet(self.obs_dim, self.action_dim, action_scale=2.)
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
        reward = torch.from_numpy(reward.astype(np.float32))
        done = self._get_done_mask(done)
        action = self._state_reformat(action, batched=True)

        v_target = self.v_net_tar(next_state)
        # update soft-Q net, use old action suite for reward to cal target
        q_0_est = self.q0_net(torch.cat([state, action], dim=-1))
        q_1_est = self.q1_net(torch.cat([state, action], dim=-1))
        q_target = reward + v_target * self.gamma * done

        pred_q0_l = tf.smooth_l1_loss(input=q_0_est, target=q_target.detach())
        pred_q1_l = tf.smooth_l1_loss(input=q_1_est, target=q_target.detach())

        self.q0_optim.zero_grad()
        pred_q0_l.backward()
        torch.nn.utils.clip_grad_norm_(self.q0_net.parameters(), self.max_grad_norm)
        self.q0_optim.step()

        self.q1_optim.zero_grad()
        pred_q1_l.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.max_grad_norm)
        self.q1_optim.step()

        # Here action is re-cal by policy to provide gradient and entropy, to this end the soft Q is cal
        action, log_prob = self.policy_net.action(state)
        q_0_est = self.q0_net(torch.cat([state, action], dim=-1))
        q_1_est = self.q1_net(torch.cat([state, action], dim=-1))
        combine_q = torch.min(q_0_est, q_1_est)
        v_cr_target = (combine_q - self.entropy_weight.detach().exp() * log_prob)
        v_cr = self.v_net(state)
        v_loss = tf.smooth_l1_loss(input=v_cr, target=v_cr_target.detach())
        self.v_optim.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), self.max_grad_norm)
        self.v_optim.step()
        # update policy
        p_l = - (combine_q - self.entropy_weight.detach().exp() * log_prob).mean()
        self.policy_optim.zero_grad()
        p_l.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optim.step()
        # update entropy weight
        e_l = self.entropy_weight * (-log_prob - self.entropy_target).detach().mean()
        self.ent_optim.zero_grad()
        e_l.backward()
        self.ent_optim.step()
        self.v_net_tar.load_state_dict(self.polyak_dict(self.v_net, self.v_net_tar, tau=self.tau))

    def learn(self, step):
        self.step_counter = 0
        while self.step_counter < step + self.learn_starts:
            state = self.env.reset()
            done = False
            while not done:
                action, log_prob = self.policy_net.action(self._state_reformat(state), deterministically=False)
                action = action.detach().cpu().numpy()[0]
                next_state, reward, done, log_info = self.env.step(action)
                if self.accumulate_reward_max < reward:
                    self.accumulate_reward_max = reward
                self.replay_buffer.push(tuple([state, action, reward, next_state, done]))
                state = next_state
                self.step_counter += 1
                if self.step_counter > self.learn_starts and self.step_counter % self.train_freq == 0:
                    for _ in range(self.gradient_steps):
                        self.model_update()

    def eval(self, epoch):
        m_reward = []
        for _ in range(epoch):
            state = self.env.reset()
            rw = 0
            for i in range(200):
                action, log_prob = self.policy_net.action(self._state_reformat(state), deterministically=True)
                action = action.detach().cpu().numpy()[0]
                next_state, reward, done, log_info = self.env.step(action)
                state = next_state
                rw += reward
                if done:
                    break
            m_reward.append(rw)
        return m_reward


if __name__ == '__main__':
    import gym
    env = gym.make('Pendulum-v1')
    model = SAC(env)
    model.learn(1e5)
    print(model.eval(10))
