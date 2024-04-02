from torch.distributions import Normal
import torch
from utils import ReplayBeffer, TargetEntropySchedule
import numpy as np
from copy import deepcopy
import gymnasium as gym


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim,
                 action_dim,
                 action_scale=1.,
                 action_bias=0, log_std_min=-20, log_std_max=2, eps=1e-6, device='cpu'):
        """Here the action dim can also be used as preform GMM model,
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
                 env_max_step_train=1000,
                 env_max_step_eval=200,
                 learning_rate=5e-4,
                 buffer_size=10000,
                 learning_starts=1000,
                 batch_size=128,
                 tau=.01,
                 gamma=0.99,
                 train_freq=4,
                 gradient_steps=1,
                 target_update_interval=1,
                 init_entropy_weight=2.,
                 max_grad_norm=10,
                 print_interval=1_000):
        self.env = env
        self.train_max_step = env_max_step_train
        self.eval_max_step = env_max_step_eval
        self.print_interval = print_interval
        self.action_dim = self.env.action_space.shape[-1]
        self.obs_dim = np.prod(self.env.observation_space.shape)

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
        self.entropy_target = TargetEntropySchedule(init_entropy=np.log(self.action_dim))
        self.ent_optim = torch.optim.Adam([self.entropy_weight], lr=self.lr)

        self.accumulate_reward_max = - np.inf
        self.q0_net = ValueNet(self.obs_dim+self.action_dim)
        self.q1_net = ValueNet(self.obs_dim+self.action_dim)
        self.q0_optim = torch.optim.Adam(self.q0_net.parameters(), lr=self.lr)
        self.q1_optim = torch.optim.Adam(self.q1_net.parameters(), lr=self.lr)

        self.q0_tar = deepcopy(self.q0_net)
        self.q1_tar = deepcopy(self.q1_net)

        self.policy_net = PolicyNet(self.obs_dim, self.action_dim)
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
        with torch.no_grad():
            action_next, log_prob_next = self.policy_net.action(next_state)
            q0_tar_next = self.q0_tar(torch.cat([next_state, action_next], dim=-1))
            q1_tar_next = self.q1_tar(torch.cat([next_state, action_next], dim=-1))
            v_tar_next = torch.min(q0_tar_next, q1_tar_next) - log_prob_next * self.entropy_weight.exp()
            q_target = self.gamma * done * v_tar_next + reward
        q0_est = self.q0_net(torch.cat([state, action], dim=-1))
        q1_est = self.q1_net(torch.cat([state, action], dim=-1))
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

        action, log_prob = self.policy_net.action(state)
        q0_est_g = self.q0_net(torch.cat([state, action], dim=-1))
        q1_est_g = self.q1_net(torch.cat([state, action], dim=-1))
        qc_est_g = torch.min(q0_est_g, q1_est_g)
        loss_p = - (qc_est_g - log_prob * self.entropy_weight.detach().exp()).mean()

        self.policy_optim.zero_grad()
        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optim.step()

        l_ent = self.entropy_weight * (-log_prob - self.entropy_target).detach().mean()
        self.ent_optim.zero_grad()
        l_ent.backward()
        self.ent_optim.step()
        self.q0_tar.load_state_dict(self.polyak_dict(self.q0_net, self.q0_tar, self.tau))
        self.q1_tar.load_state_dict(self.polyak_dict(self.q1_net, self.q1_tar, self.tau))

    def learn(self, step):
        self.step_counter = 0
        while self.step_counter < step + self.learn_starts:
            state = self.env.reset()
            for _ in range(self.train_max_step):
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
                if self.step_counter % self.print_interval == 0:
                    print(self.eval(5))

    def eval(self, epoch):
        m_reward = []
        for _ in range(epoch):
            state = self.env.reset()
            rw = 0
            for _ in range(self.eval_max_step):
                action, log_prob = self.policy_net.action(self._state_reformat(state), deterministically=True)
                action = action.detach().cpu().numpy()[0]
                next_state, reward, done, log_info = self.env.step(action)
                state = next_state
                rw += reward
                if done:
                    break
            m_reward.append(np.sum(rw))
        return np.mean(m_reward)


class Walker(gym.Env):
    def __init__(self, bins=20):
        self.env = gym.make("BipedalWalker-v3")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.MultiDiscrete([bins]*self.env.action_space.shape[0])
        self.discrete_action = np.linspace(-1., 1., bins)

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        return next_state, reward, done, info

    def reset(self):
        next_state = self.env.reset()[0]
        return next_state

    def render(self, mode="human"):
        self.env.render(mode=mode)

    def seed(self, seed=None):
        self.env.seed(seed)


if __name__ == '__main__':
    env = Walker()
    model = SAC(env)
    model.learn(1e6)
    print(model.eval(20))
