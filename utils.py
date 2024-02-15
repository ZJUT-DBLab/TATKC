import numpy as np


# Utility function and class
import torch
from scipy.stats import kendalltau


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


def loss_cal(y_out, true_val, num_nodes, device):
    _, order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes * 20

    ind_1 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)
    ind_2 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)

    rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()

    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)

    loss_rank = torch.nn.MarginRankingLoss(margin=1).forward(input_arr1, input_arr2, rank_measure)

    return loss_rank
