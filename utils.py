import numpy as np
import torch
import torch.nn.functional as tf
import collections
import random


def one_hot(data, depth):
    data_shape = data.shape
    data = data.reshape([-1])
    res = np.zeros([data.shape[0], depth], dtype=np.float32)
    for i, c in enumerate(data):
        res[i, c] = 1
    res = res.reshape(list(data_shape) + [depth])
    return res


def get_cellular_norm(nr_cell):
    """
    Create a cellular topology whit normalized ele2ele distance
    :param nr_cell:
    :return:
    """
    cell_pos = None
    init_row_num = nr_cell // 2 + 1
    for i in range(nr_cell // 2 + 1):
        if cell_pos is None:
            temp = np.concatenate([np.arange(init_row_num)[:, None], np.zeros([init_row_num, 1])], axis=-1)
            center = np.asarray([nr_cell // 2 / 2, nr_cell // 2 * np.sqrt(3) / 2])
            temp -= center
            cell_pos = [temp]

        else:
            local_pos = cell_pos[-1].copy()
            local_pos[..., 0] += 1 / 2
            local_pos[..., 1] += np.sqrt(3) / 2
            temp = local_pos[0:1].copy()
            temp[:, 0] -= 1
            local_pos = np.concatenate([temp, local_pos], axis=0)
            cell_pos.append(local_pos)
    inv_pos = np.concatenate(cell_pos[:-1][::-1], axis=0)
    inv_pos[:, -1] *= -1
    cell_pos.append(inv_pos)
    cell_pos = np.concatenate(cell_pos, axis=0)
    return cell_pos


def get_norm_illumination_coord(norm_cellular, r_earth, sat_height, theta_3db, expand_ratio=np.sqrt(3)):
    theta_3db = theta_3db / 2 / 180 * np.pi
    phis = np.arctan2(norm_cellular[..., 0], norm_cellular[..., 1])
    thetas = np.linalg.norm(norm_cellular, axis=-1) * theta_3db * expand_ratio
    a = 1
    b = -2 * (sat_height + r_earth) * np.cos(thetas)
    c = (sat_height + r_earth) ** 2 - r_earth ** 2
    d = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2 / a
    r = d * np.sin(thetas)
    z = r_earth + sat_height - d * np.cos(thetas)
    cell_pos = np.stack([r * np.cos(phis), r * np.sin(phis), z], axis=-1)
    return cell_pos


def get_rotation_matrix(theta=None, phi=None, beta=None):
    """
    positive angle means clock-wise rotation
    :param theta: refers  angle to rotate around the z-axis
    :param phi: angle to rotate around the y-axis
    :param beta: angle to rotate around the x-axis
    :return:
    """
    ret_theta = np.eye(3)
    if theta is not None:
        ret_theta = ret_theta @ np.asarray([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])
    if phi is not None:
        ret_theta = ret_theta @ np.asarray([[np.cos(phi), 0, np.sin(phi)],
                                            [0, 1, 0],
                                            [-np.sin(phi), 0, np.cos(phi)]])
    if beta is not None:
        ret_theta = ret_theta @ np.asarray([[1, 0, 0],
                                            [0, np.cos(beta), -np.sin(beta)],
                                            [0, np.sin(beta), np.cos(beta)]])
    return ret_theta


def gumbel_softmax(logits, tau=1, require_prob=False, dim=-1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    log_prob = tf.log_softmax(gumbels, dim=dim)
    y_soft = log_prob.exp()
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = (y_hard - y_soft).detach() + y_soft
    if require_prob:
        return ret, y_soft, log_prob
    else:
        return ret


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


class TargetEntropySchedule:
    def __init__(self, init_entropy, discount=.9, decay_steps=20, lambda_=.999, avg_t=1e-2, var_t=5e-2):
        super(TargetEntropySchedule, self).__init__()
        self.lambda_ = lambda_
        self.decay_steps = decay_steps
        self.discount = discount
        self.mu = init_entropy
        self.mu_t = avg_t
        self.sigma = 0.
        self.sigma_t = var_t
        self.step_counter = 0
        self.cr_target_entropy = init_entropy

    def step(self, log_porbs):
        with torch.no_grad():
            e_t = torch.mean(-log_porbs)
            if self.mu is None and self.cr_target_entropy is None:
                max_entropy = torch.max(-log_porbs)
                self.mu, self.cr_target_entropy = max_entropy, max_entropy
            delta = e_t - self.mu
            self.mu = self.mu + (1 - self.lambda_) * delta
            sigma_2 = self.lambda_ * (self.sigma**2 + (1 - self.lambda_) * delta**2)
            self.sigma = torch.sqrt(sigma_2)
            low_bound = self.cr_target_entropy - self.mu_t
            high_bound = self.cr_target_entropy + self.mu_t
            cond = not (low_bound < self.mu < high_bound) or (self.sigma > self.sigma_t)
            if cond:
                return self.cr_target_entropy
            self.step_counter += 1
            if self.step_counter > self.decay_steps:
                self.step_counter = 0
                self.cr_target_entropy *= self.discount
                return self.cr_target_entropy
