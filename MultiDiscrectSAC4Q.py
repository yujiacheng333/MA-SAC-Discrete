from utils import one_hot
from torch.distributions import Categorical
import torch
import collections
import random
import numpy as np
from copy import deepcopy
import gymnasium as gym


class Walker(gym.Env):
    def __init__(self, bins=20):
        self.env = gym.make("BipedalWalker-v3")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.MultiDiscrete([bins] * self.env.action_space.shape[0])
        # self.discrete_action = np.linspace(-1., 1., bins)

    def step(self, action):
        # continuous_action = self.discrete_action[action]
        next_state, reward, done, _, info = self.env.step(action)
        return next_state, reward, done, info

    def reset(self):
        next_state = self.env.reset()[0]
        return next_state

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def seed(self, seed=None):
        self.env.seed(seed)


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
    def __init__(self, state_dim, action_dim, repeat_action, ):
        super(PolicyNet, self).__init__()
        self.action_dim = action_dim
        self.repeat_action = repeat_action
        self.model = torch.nn.Sequential(*[torch.nn.Linear(state_dim, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, repeat_action * action_dim)])

    def forward(self, x):
        temp = self.model(x).reshape([-1, self.repeat_action, self.action_dim])
        return torch.nn.functional.log_softmax(temp, dim=-1)

    def action(self, state, deterministically=False, require_log_prob=False):
        log_prob = self(state)
        prob = log_prob.exp()
        if deterministically:
            discrete_action = torch.argmax(prob, dim=-1)
        else:
            discrete_action = Categorical(prob).sample().squeeze()
        if require_log_prob:
            return discrete_action, log_prob
        else:
            return discrete_action


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


class Qnet(torch.nn.Module):
    """
    Q(action, state), used to cal env-action pair score
    """

    def __init__(self, state_dim, repeat_action, action_dim):
        super(Qnet, self).__init__()
        self.action_dim = action_dim
        self.repeat_action = repeat_action
        self.model = torch.nn.Sequential(*[torch.nn.Linear(state_dim, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, action_dim * repeat_action)])

    def forward(self, x):
        x = self.model(x)
        return x.reshape([-1, self.repeat_action, self.action_dim])


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
                 buffer_size=10000,
                 learning_starts=1000,
                 batch_size=128,
                 tau=.01,
                 gamma=0.99,
                 train_freq=4,
                 gradient_steps=1,
                 target_update_interval=1,
                 init_entropy_weight=0.,
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

        self.repeat_action = len(self.env.action_space.nvec)
        self.action_dim = self.env.action_space.nvec[0]
        self.obs_dim = np.prod(self.env.observation_space.shape)

        self.entropy_weight = torch.ones(1) * init_entropy_weight
        self.entropy_weight.requires_grad = True
        self.entropy_target = 0.2 * np.log(self.action_dim)
        self.ent_optim = torch.optim.Adam([self.entropy_weight], lr=self.lr)

        self.step_counter = 0

        self.accumulate_reward_max = - np.inf

        self.q0_net = Qnet(state_dim=self.obs_dim,
                           action_dim=self.action_dim,
                           repeat_action=self.repeat_action)
        self.q1_net = Qnet(state_dim=self.obs_dim,
                           action_dim=self.action_dim,
                           repeat_action=self.repeat_action)

        self.q0_tar = deepcopy(self.q0_net)
        self.q1_tar = deepcopy(self.q1_net)

        self.policy_net = PolicyNet(self.obs_dim, self.action_dim, repeat_action=4)

        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q0_optim = torch.optim.Adam(self.q0_net.parameters(), lr=self.lr)
        self.q1_optim = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr)
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

    def cal_combine_action(self, multi_disc, repeat_dim=1):
        if not isinstance(multi_disc, np.ndarray):
            multi_disc = multi_disc.numpy()
        multi_disc = one_hot(multi_disc, depth=self.action_dim).sum(repeat_dim)
        return torch.from_numpy(multi_disc)

    def model_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = self._state_reformat(state, batched=True)
        action_one_hot = torch.from_numpy(one_hot(action, depth=self.action_dim))
        next_state = self._state_reformat(next_state, batched=True)
        reward = torch.from_numpy(reward.astype(np.float32))
        done = self._get_done_mask(done)
        with torch.no_grad():
            action_next, log_prob_next = self.policy_net.action(next_state, require_log_prob=True)
            prob_next = log_prob_next.exp()
            q0_tar_next = self.q0_tar(next_state)
            q1_tar_next = self.q1_tar(next_state)
            v_next = prob_next * (torch.min(q0_tar_next, q1_tar_next)
                                  - self.entropy_weight.detach().exp() * log_prob_next)
            v_next = v_next.sum(-1).mean(-1)
            q_target = self.gamma * done * v_next + reward
        q0_ret = self.q0_net(state)
        q1_ret = self.q1_net(state)
        q0_est = (q0_ret * action_one_hot).sum(-1).mean(-1)
        q1_est = (q1_ret * action_one_hot).sum(-1).mean(-1)
        loss_q0 = torch.nn.functional.smooth_l1_loss(q0_est, q_target.detach())
        loss_q1 = torch.nn.functional.smooth_l1_loss(q1_est, q_target.detach())

        self.q0_optim.zero_grad()
        loss_q0.backward()
        torch.nn.utils.clip_grad_norm_(self.q0_net.parameters(), self.max_grad_norm)
        self.q0_optim.step()

        self.q1_optim.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), self.max_grad_norm)
        self.q1_optim.step()
        combine_q = torch.min(q0_ret, q1_ret).detach()
        action, log_prob = self.policy_net.action(state, require_log_prob=True)
        prob = log_prob.exp()
        loss_p = - (prob * (combine_q.detach() - self.entropy_weight.detach().exp() * log_prob)).sum(-1).mean()
        self.policy_optim.zero_grad()
        loss_p.backward()
        loss_ent = - self.entropy_weight * (prob * (log_prob + self.entropy_target)).detach().sum(-1).mean()
        self.ent_optim.zero_grad()
        loss_ent.backward()
        self.ent_optim.step()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optim.step()
        self.q0_tar.load_state_dict(self.polyak_dict(self.q0_net, self.q0_tar, self.tau))
        self.q1_tar.load_state_dict(self.polyak_dict(self.q1_net, self.q1_tar, self.tau))

    def learn(self, step):
        self.step_counter = 0
        while self.step_counter < step + self.learn_starts:
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy_net.action(self._state_reformat(state), deterministically=False)
                action = action.detach().cpu().numpy()
                next_state, reward, done, log_info = self.env.step(action)
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
                if self.step_counter % 1000 == 0:
                    print(self.eval(5), self.entropy_weight)

    def eval(self, epoch):
        m_reward = []
        for _ in range(epoch):
            state = self.env.reset()
            lw = 0.
            for i in range(200):
                action = self.policy_net.action(self._state_reformat(state), deterministically=True)
                action = action.squeeze().detach().cpu().numpy()
                next_state, reward, done, log_info = self.env.step(action)
                state = next_state
                lw += reward
                if done:
                    break
            m_reward.append(lw)
        return np.mean(m_reward)


if __name__ == '__main__':
    env = Walker()
    model = SAC(env)
    model.learn(2e6)
    print(model.eval(20))
