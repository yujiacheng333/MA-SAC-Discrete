import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import gymnasium as gym
"""
PPO with on line training framework
Not completed！！！！！！！！！！！！！
Do not use it！！！！！！！！！！！！！
"""

class RollOutBuffer:
    def __init__(self):
        super().__init__()
        self.item = {}
        self.end_item = {}

    def push(self, **kwargs):
        for k in kwargs.keys():
            try:
                self.item[k].append(kwargs[k])
            except KeyError:
                print('New key {} is updated in RoBuffer'.format(k))
                self.item.update({k: [kwargs[k]]})

    def clear(self):
        for k in self.item:
            self.item[k] = []


class FCNBase(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(FCNBase, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.model(x)


class PolicyNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_scale=1.,
                 action_bias=0, log_std_min=-20, log_std_max=2, eps=1e-6,
                 ):
        super(PolicyNet, self).__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.eps = eps
        self.model = FCNBase(state_dim, action_dim * 2)

    def forward(self, x):
        temp = self.model(x)
        temp = temp.view(-1, self.action_dim, 2)
        mean, log_std = temp[..., 0], temp[..., 1]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        distribute = Normal(mean, torch.exp(log_std))
        return distribute

    def action(self, state, y=None):
        distribute = self(state)
        if y is None:
            x = distribute.rsample()
            log_px = distribute.log_prob(x)
            y = torch.tanh(x) * self.action_scale + self.action_bias
        else:
            x = (y - self.action_bias) / self.action_scale
            x = torch.arctanh(x)
            log_px = distribute.log_prob(x)
        log_dx_dy = np.log(self.action_scale) - torch.log(self.action_scale ** 2 - (y - self.action_bias) ** 2)
        log_py = log_px + log_dx_dy
        return y, log_py.sum(-1)


class PolicyNetDiscrete(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim):
        super(PolicyNetDiscrete, self).__init__()

        if isinstance(action_dim, int):
            self.action_dim = action_dim
            self.model = FCNBase(state_dim, action_dim)
            self.multi_action = False
        elif isinstance(action_dim, list):
            self.action_dim = action_dim
            self.model = FCNBase(state_dim, np.sum(action_dim))
            self.multi_action = True
        else:
            raise ValueError

    def forward(self, x):
        temp = self.model(x)
        if self.multi_action:
            log_prob = torch.split(temp, self.action_dim, dim=-1)
            log_prob = [F.softmax(i, dim=-1) for i in log_prob]
        else:
            log_prob = F.softmax(temp, dim=-1)
        return log_prob

    def action(self, x, action=None, require_entropy=False):
        prob = self(x)
        if self.multi_action:
            n_action = len(prob)
            distribute = [Categorical(probs=prob[i]) for i in range(n_action)]
            if action is None:
                action = [distribute[i].sample() for i in range(n_action)]
            log_prob = 0
            entropy = 0
            for i in range(n_action):
                log_prob += distribute[i].log_prob(action[i])
                entropy += distribute[i].entropy()
        else:
            distribute = Categorical(probs=prob)
            action = distribute.sample()
            log_prob = distribute.log_prob(action)
            entropy = distribute.entropy()
        if not require_entropy:
            return action, log_prob
        else:
            return action, log_prob, entropy


class PPO:
    def __init__(self,
                 env_id='CartPole-v1',
                 num_vec_env=8,
                 max_env_step=200,
                 lr=5e-4,
                 max_grd_norm=5.,
                 gamma=0.99,
                 lam=.5,
                 eps=1e-6, batch_size=32,
                 clip_coef=.1,
                 entropy_coeff=0.01):
        self.clip_value_loss = True
        self.batch_size = batch_size
        self.vec_env = gym.vector.make(env_id,
                                       num_envs=num_vec_env)
        self.num_vec_env = num_vec_env
        self.max_env_step = max_env_step
        self.gammma = gamma
        self.lam = lam
        self.lr = lr
        self.max_grd_norm = max_grd_norm
        self.state_dim = self.vec_env.single_observation_space.shape[0]
        self.action_dim = int(self.vec_env.single_action_space.n)
        self.eps = eps
        self.ro_buffer = RollOutBuffer()
        self.policy_net = PolicyNetDiscrete(state_dim=self.state_dim,
                                            action_dim=self.action_dim)
        self.optimizer_p = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_net = FCNBase(self.state_dim, 1)
        self.optimizer_v = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coeff

    @staticmethod
    def _state_reformat(state, batched=False):
        state = state.astype(np.float32)
        state = state.reshape([1, -1]) if not batched else state.reshape([state.shape[0], -1])
        state = torch.from_numpy(state)
        return state

    def collect_roll_out(self):
        with torch.no_grad():
            state = self.vec_env.reset()[0]
            next_state = None
            for i in range(self.max_env_step):
                state = self._state_reformat(state, batched=True)
                action, log_prob = self.policy_net.action(state)
                value = self.value_net(state).squeeze()
                next_state, reward, term, _, info = self.vec_env.step(action.cpu().numpy())
                self.ro_buffer.push(state=state,
                                    value=value,
                                    action=action,
                                    log_prob=log_prob,
                                    reward=reward,
                                    term=term,
                                    info=info)
                state = next_state
            self.ro_buffer.end_item.update({'end_state': next_state})

    def ro_preprocess(self, gae=True):
        self.ro_buffer.end_item['end_state'] = self._state_reformat(self.ro_buffer.end_item['end_state'], batched=True)
        self.ro_buffer.item['state'] = torch.stack(self.ro_buffer.item['state'], dim=1)
        self.ro_buffer.item['reward'] = torch.from_numpy(np.asarray(self.ro_buffer.item['reward'], dtype=np.float32).T)
        self.ro_buffer.item['value'] = torch.stack(self.ro_buffer.item['value'], dim=1)
        self.ro_buffer.item['action'] = torch.stack(self.ro_buffer.item['action'], dim=1)
        self.ro_buffer.item['log_prob'] = torch.stack(self.ro_buffer.item['log_prob'], dim=1)
        self.ro_buffer.item['term'] = torch.from_numpy(np.asarray(self.ro_buffer.item['term'], dtype=np.float32)).T

        with torch.no_grad():

            v_next = self.value_net(self.ro_buffer.end_item['end_state']).squeeze()
            if gae:
                advance = 0.
                advantages = []
                for i_step in reversed(range(self.max_env_step)):
                    local_not_term = 1. - self.ro_buffer.item['term'][:, i_step]
                    discount_value = self.gammma * v_next * local_not_term
                    q = discount_value + self.ro_buffer.item['reward'][:, i_step]
                    delta = q - self.ro_buffer.item['value'][:, i_step]
                    advance = delta + self.gammma * self.lam * local_not_term * advance
                    advantages.append(advance)
                    v_next = self.ro_buffer.item['value'][:, i_step]
                advantages = torch.stack(advantages[::-1], dim=1)
                returns = advantages + self.ro_buffer.item['value']
            else:
                returns = []
                _return = v_next
                for i_step in reversed(range(self.max_env_step)):
                    local_not_term = 1. - self.ro_buffer.item['term'][:, i_step]
                    _return = self.ro_buffer.item['reward'] + self.gammma * local_not_term * _return
                    returns.append(_return)
                returns = torch.stack(returns[::-1], dim=1)
                advantages = returns - self.ro_buffer.item['value']
        self.ro_buffer.item.update({'returns': returns})
        self.ro_buffer.item.update({'advantages': advantages})

    def flatten_ro_buffer(self):
        for k in self.ro_buffer.item.keys():
            if isinstance(self.ro_buffer.item[k], torch.Tensor):
                local_shape = self.ro_buffer.item[k].shape
                target_shape = [-1] + list(local_shape[2:]) if len(local_shape) > 2 else [-1]
                self.ro_buffer.item[k] = torch.reshape(self.ro_buffer.item[k], target_shape)
            else:
                print('Key {} is not instance of torch.Tensor, passed'.format(k))

    def update_agents(self, epoches):
        step_per_epoch = np.maximum((self.max_env_step * self.num_vec_env) // self.batch_size, 1)
        for e in range(epoches):
            sampler = np.arange(self.max_env_step * self.num_vec_env)
            np.random.shuffle(sampler)
            for s in range(step_per_epoch):
                local_sampler = sampler[s * self.batch_size:(s + 1) * self.batch_size]
                batch_state = self.ro_buffer.item['state'][local_sampler]
                batch_action = self.ro_buffer.item['action'][local_sampler]
                batch_log_prob = self.ro_buffer.item['log_prob'][local_sampler]
                batch_return = self.ro_buffer.item['returns'][local_sampler]
                batch_value = self.ro_buffer.item['value'][local_sampler]
                _, new_batch_log_prob, entropy = self.policy_net.action(x=batch_state,
                                                                        action=batch_action,
                                                                        require_entropy=True)
                log_prob_ratio = new_batch_log_prob - batch_log_prob.detach()
                prob_ratio = log_prob_ratio.exp()
                with torch.no_grad():
                    old_app_kl = (-log_prob_ratio).mean()
                    local_app_kl = ((prob_ratio - 1.) - log_prob_ratio).mean()
                batch_advantage = self.ro_buffer.item['advantages'][local_sampler]
                batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
                # nature
                loss_policy_n = - batch_advantage * prob_ratio
                # clip
                loss_policy_c = - batch_advantage * torch.clamp(prob_ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                loss_policy = torch.max(loss_policy_n, loss_policy_c).mean()
                entropy_loss = - entropy.mean()
                loss_policy += entropy_loss * self.entropy_coef

                self.optimizer_p.zero_grad()
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grd_norm)
                self.optimizer_p.step()

                new_value = self.value_net(batch_state).squeeze()
                if self.clip_value_loss:
                    v_loss = (new_value - batch_return)**2
                    clip_v = torch.clamp(new_value, min=batch_value-self.clip_coef, max=batch_value+self.clip_coef)
                    clip_v_loss = (clip_v - batch_return) ** 2
                    v_loss = torch.max(v_loss, clip_v_loss)
                    v_loss = 0.5 * v_loss.mean()
                else:
                    v_loss = (new_value - batch_return)**2
                self.optimizer_v.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grd_norm)
                self.optimizer_v.step()
        return old_app_kl, local_app_kl

    def evaluate(self, n_times=1):
        # TODO: Not completed
        with torch.no_grad():
            m_reward = []
            for _ in range(n_times):
                state = self.vec_env.reset()[0]
                for i in range(self.max_env_step):
                    state = self._state_reformat(state, batched=True)
                    action, log_prob = self.policy_net.action(state)
                    next_state, reward, term, _, info = self.vec_env.step(action.cpu().numpy())
                    state = next_state
                    m_reward.append(reward)
            print(np.mean(m_reward))

    def learn(self, num_update=1):
        for i in range(num_update):
            self.collect_roll_out()
            self.ro_preprocess()
            self.flatten_ro_buffer()
            self.update_agents(num_update)
            self.ro_buffer.clear()
            self.evaluate()


if __name__ == '__main__':
    ppo = PPO(max_env_step=200)
    ppo.learn(50)
