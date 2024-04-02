import matplotlib.pyplot as plt
import torch.nn
from copy import deepcopy
from utils import *
import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.buffers import ReplayBuffer

np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
C = 3e8
EPS = 1e-9
R_EARTH = 6371e3
GEO_HEIGHT = 36000e3
SAT_HEIGHT = 35786e3
KB = 1.3806e-23
Noise_T = 354


class GeoRsmaEnv:
    def __init__(self):
        super(GeoRsmaEnv, self).__init__()
        """
        can never exceeded TTL
        """
        self.step_counter = 0
        self.TTL = 40
        self.sat_pos_vec = np.asarray([0, 0, SAT_HEIGHT + R_EARTH])
        self.nr_cell = 6
        norm_cellular = get_cellular_norm(self.nr_cell)
        self.effective_beam_width = 1.5  # in degree
        self.fc = 20.5e9
        self.transmit_power = 34  # in dBW = 3000W
        self.beam_hopping_window = 1.3  # in ms
        self.len_time_slot = self.beam_hopping_window / 256
        self.ut_gain = 32.55
        self.bw = 500e6
        self.illumination_center = get_norm_illumination_coord(norm_cellular,
                                                               R_EARTH,
                                                               GEO_HEIGHT,
                                                               self.effective_beam_width)
        self.num_cell = self.illumination_center.shape[0]
        self.num_beam_per_slot = 4
        self.demand_per_cell = 100
        self.ut_number_per_cell = 4
        self.demand = None
        self.gain_numpy_func = None

    def reset(self):
        mean_demand_u = self.demand_per_cell / self.ut_number_per_cell
        self.demand = np.zeros([self.TTL, self.num_cell, self.ut_number_per_cell])
        self.demand[0] = np.random.poisson(mean_demand_u, size=[self.num_cell, self.ut_number_per_cell])
        self.step_counter = 0
        return self.demand

    def add_new_demand(self):
        mean_demand_u = self.demand_per_cell / self.ut_number_per_cell
        self.demand[1:] = self.demand[:-1]
        self.demand[0] = np.random.poisson(mean_demand_u, size=[self.num_cell, self.ut_number_per_cell])

    def ant_radiation_pattern(self, delta_theta, a=2.88, b=6.32, gm=41.6, ls=-25):
        """
        Sat Ant Gain proposed in ITU-S.672 (c) protocol
        :param delta_theta: off-center angle (with abs), note in degree
        :param theta_b: 3dB beam width, note in degree
        :param a: Gain function param
        :param b:Gain function param
        :param gm:max Gain of sat ant
        :param ls: the required near in side lobe level (旁瓣点平等级)
        :return: Response of SAT beam Gain in dBi
        """
        theta_b = self.effective_beam_width / 2
        if delta_theta < theta_b:
            return gm
        elif delta_theta < a * theta_b:
            return gm - 3 * (delta_theta / theta_b) ** 2
        elif delta_theta < b * theta_b:
            return gm + ls
        else:
            return np.maximum(gm + ls + 20 - 25 * np.log10(delta_theta / theta_b), 0)

    @staticmethod
    def cal_path_loss(propagation_dist, carrier_freq):
        # """
        # 3GPP propagation loss
        # for narrow band system carrier freq can be replaced by center freq!
        # """
        # return 32.4 + 20 * np.log10(propagation_dist) + 20 * np.log10(carrier_freq / 1e6)
        return 20 * np.log10(4 * np.pi * propagation_dist * carrier_freq / 3e8)

    def get_cross_beam_gain(self, direction_vectors):
        if self.gain_numpy_func is None:
            self.gain_numpy_func = np.frompyfunc(lambda x: self.ant_radiation_pattern(x), nin=1, nout=1)
        cross_angle_map = direction_vectors / np.linalg.norm(direction_vectors, axis=-1, ord=2, keepdims=True)
        cross_angle_map = np.clip(cross_angle_map @ cross_angle_map.T, a_min=-1 + EPS, a_max=1 - EPS)
        cross_angle_map = np.arccos(cross_angle_map) / np.pi * 180
        cross_beam_gain = self.gain_numpy_func(cross_angle_map)
        return cross_beam_gain

    def mmse_precodding(self, heq):
        eye_mat = np.eye(heq.shape[0], dtype=complex)
        weight = heq.conj().T @ np.linalg.inv(heq @ heq.conj().T + eye_mat)
        return weight / np.linalg.norm(weight, ord=2)

    # def zf_precodding(self, heq):
    #     eye_mat = np.eye(heq.shape[0], dtype=np.complex)
    #     power_diag = eye_mat * np.diag(np.abs(heq) ** 2)
    #     weight = power_diag @ heq.conj().T @ np.linalg.inv(heq @ power_diag @ heq.conj().T)
    #     return weight / np.linalg.norm(weight, axis=0, keepdims=True, ord=2)
    #
    # def no_precodding(self, heq):
    #     return np.eye(heq.shape[0], dtype=np.complex)

    def step(self, active_cell, precoding_weight):
        assert len(active_cell) == self.num_beam_per_slot
        beam_direction_vectors = self.illumination_center[active_cell] - self.sat_pos_vec[None]
        cross_beam_gain = self.get_cross_beam_gain(beam_direction_vectors)
        dist = np.linalg.norm(beam_direction_vectors, axis=-1, ord=2)
        cross_path_loss = self.cal_path_loss(dist, self.fc)
        rssi = self.transmit_power - 10 * np.log10(self.num_beam_per_slot) + \
               cross_beam_gain - cross_path_loss + self.ut_gain
        noise_power = 10 * np.log10(self.bw * KB * Noise_T)
        normalized_signal_amp = 10 ** ((rssi - noise_power) / 20.)
        normalized_channel = normalized_signal_amp.astype(complex)
        if precoding_weight == 'mmse':
            precoding_weight = self.mmse_precodding(normalized_channel)
        elif precoding_weight == 'no':
            precoding_weight = np.eye(self.num_beam_per_slot)
        else:
            raise NotImplementedError('没做呢！')
        beam_reponse = np.abs(precoding_weight @ normalized_channel) ** 2
        recv_power = np.sum(beam_reponse * np.eye(self.num_beam_per_slot), axis=0)
        inter_power = np.sum(beam_reponse * (1 - np.eye(self.num_beam_per_slot)), axis=0)
        sinr = recv_power / (inter_power + 1)
        achievable_rate = np.log2(1 + sinr) * self.bw / 1e6  # in MHz
        thoughput = 0.
        for idx, cid in enumerate(active_cell):
            local_rate = achievable_rate[idx].copy()
            for t in range(self.demand.shape[0] - 1, -1, -1):
                if local_rate == 0:
                    break
                elif self.demand[t, cid].sum() != 0:
                    for u in range(self.demand.shape[-1]):
                        if self.demand[t, cid, u] <= local_rate:
                            local_rate -= self.demand[t, cid, u]
                            thoughput += self.demand[t, cid, u]
                            self.demand[t, cid, u] = 0.
                        else:
                            self.demand[t, cid, u] -= local_rate
                            thoughput += local_rate
                            local_rate = 0
                            break
        time_delay = np.einsum('tcu, t->cu', self.demand, np.arange(1, self.TTL + 1))
        sum_demand = np.sum(self.demand, axis=0)
        time_delay = time_delay / sum_demand
        time_delay[np.isnan(time_delay)] = 0.
        fairness = np.max(time_delay) - np.min(time_delay)
        mean_delay = np.mean(time_delay)
        self.add_new_demand()
        self.step_counter += 1
        return self.demand, thoughput, fairness, mean_delay, self.step_counter


class CustomGymEnv(gym.Env):
    def __init__(self):
        """多继承太麻烦了，直接实例化一个，反正不影响！
        而且reward计算可以放到这边来，更方便！
        """
        super(CustomGymEnv, self).__init__()
        self.geo_env = GeoRsmaEnv()
        self.TTL = self.geo_env.TTL
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=[self.geo_env.TTL,
                                                   self.geo_env.num_cell])
        self.action_space = spaces.MultiDiscrete([self.geo_env.num_cell for _ in range(self.geo_env.num_beam_per_slot)])

    def step(self, action):
        """observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)"""
        demand, thoughput, fairness, mean_delay, step = self.geo_env.step(action, 'no')
        demand = demand.sum(-1)
        demand /= self.geo_env.demand_per_cell
        observation = demand
        reward = thoughput / 1000. / self.geo_env.num_beam_per_slot \
                 - (fairness + mean_delay) / self.geo_env.TTL * 2.
        info = {"demand": demand,
                "thoughput": thoughput,
                "fairness": fairness,
                "mean_delay": mean_delay}
        return observation, reward, step > self.geo_env.TTL - 1, info

    def reset(self):
        return self.geo_env.reset().sum(-1) / self.geo_env.demand_per_cell

    def close(self):
        super(CustomGymEnv, self).close()

    def seed(self, seed=None):
        super(CustomGymEnv, self).seed(seed)

    def render(self, mode="human"):
        super(CustomGymEnv, self).render(mode)


if __name__ == '__main__':
    from stable_baselines3.ppo import PPO
    env = CustomGymEnv()
    # model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]}, gamma=.99,
    #             batch_size=128, learning_rate=5e-4)
    # model.learn(total_timesteps=1e5, log_interval=4)
    # model.save("ppo_multi-beam")
    model = PPO.load("ppo_multi-beam")
    for i in range(40):
        m_reward = 0
        state = env.reset()
        done = False
        while not done:
            action, _s = model.predict(state)
            state, r, done, info = env.step(action)
            m_reward += r
        print(m_reward)
