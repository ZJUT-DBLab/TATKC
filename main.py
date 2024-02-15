import math
import logging
import pickle
import time
import random
import sys
import argparse

import scipy
import torch
import pandas as pd
import numpy as np

from scipy.stats import kendalltau
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm


from module import TATKC
from nx2graphs import load_real_data, load_real_true_TKC, load_train_real_data, load_real_train_true_TKC
from utils import loss_cal


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 64)
        self.fc_2 = torch.nn.Linear(64, 32)
        self.fc_3 = torch.nn.Linear(32, 1)

        self.act = torch.nn.ReLU()

        torch.nn.init.kaiming_normal_(self.fc_1.weight)
        torch.nn.init.kaiming_normal_(self.fc_2.weight)
        torch.nn.init.kaiming_normal_(self.fc_3.weight)

        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


# Argument and global variables
parser = argparse.ArgumentParser('Interface for TATKC experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-tgwiktioanry')
parser.add_argument('--bs', type=int, default=1500, help='batch_size')
parser.add_argument('--prefix', type=str, default='hello_world', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=3, help='idx for the gpu to use')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument("--local_rank", type=int)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(1)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
LR_MODEL_SAVE_PATH = f'./saved_models/{args.agg_method}-{args.attn_mode}-{args.data}_mlp.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def eval_real_data(hint, tgan, lr_model, sampler, src, ts, label):
    val_acc, val_kts = [], []
    test_pred_tbc_list = []
    tgan.ngh_finder = sampler

    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])
            src_embed = tgan.tem_conv(test_src_l_cut, test_ts_l_cut, NUM_LAYER, num_neighbors=NUM_NEIGHBORS)
            test_pred_tbc = lr_model(src_embed)
            test_pred_tbc_list.extend(test_pred_tbc.cpu().detach().numpy().tolist())

        kt, _ = kendalltau(test_pred_tbc_list, label)
        numslist = [0.01, 0.05, 0.1, 0.2]
        acc_list = []
        for k in numslist:
            nums = int(k * len(src))
            test_pred_tbc_topk = np.argsort(test_pred_tbc_list)[-nums:]
            test_true_data_label_topk = np.argsort(label)[-nums:]
            test_hit = list(set(test_pred_tbc_topk).intersection(set(test_true_data_label_topk)))
            val_acc_topk = min((len(test_hit) / nums), 1.00)
            acc_list.append(val_acc_topk)
        val_kts.append(kt)

    return acc_list, np.mean(val_kts)


# Load data
n_feat = np.load('./data/test/Real/processed/ml_{}_node.npy'.format(DATA), allow_pickle=True)
# test_real_feat = np.load('./data/test/Real/processed/ml_{}_node.npy'.format(DATA), allow_pickle=True)
test_real_feat = np.zeros((4200000, 128))


def setSeeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


setSeeds(89)


train_real_src_l, train_real_dst_l, train_real_ts_l, train_real_node_count, train_real_node, train_real_time, \
    train_real_ngh_finder = load_train_real_data(UNIFORM)

test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, test_real_node, test_real_time, \
    test_real_ngh_finder = load_real_data(dataName=DATA)


nodeList_train_real, train_label_l_real = load_real_train_true_TKC()
nodeList_test_real, test_label_l_real = load_real_true_TKC('{}'.format(DATA))
train_ts_list, test_ts_list, train_real_ts_list = [], [], []

for idx in range(len(nodeList_train_real)):
    train_real_ts_list.append(np.array([train_real_time[idx]] * len(nodeList_train_real[idx])))

test_real_ts_list = np.array([test_real_time] * len(nodeList_test_real))

TEST_BATCH_SIZE = BATCH_SIZE
num_test_instance = len(nodeList_test_real)
num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
for k in range(num_test_batch):
    s_idx = k * TEST_BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
    test_src_l_cut = np.array(nodeList_test_real[s_idx:e_idx])
    test_ts_l_cut = np.array(test_real_ts_list[s_idx:e_idx])
    test_real_ngh_finder.preprocess(tuple(test_src_l_cut), tuple(test_ts_l_cut), NUM_LAYER, NUM_NEIGHBORS)

# Model initialize
device = torch.device('cuda:{}'.format(GPU))

MLP_model = MLP(n_feat.shape[1], drop=DROP_OUT)
MLP_model = MLP_model.to(device)
tatkc = TATKC(train_real_ngh_finder[0], test_real_feat, num_layers=NUM_LAYER, use_time=USE_TIME,
              agg_method=AGG_METHOD, attn_mode=ATTN_MODE, seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT)
tatkc = tatkc.to(device)

optimizer = torch.optim.Adam(list(tatkc.parameters()) + list(MLP_model.parameters()), LEARNING_RATE)

tatkc.load_state_dict(torch.load('./saved_models/model_tatkc.pth'))
MLP_model.load_state_dict(torch.load('./saved_models/model_MLP.pth'))

print('train start')
start_train = time.time()
for epoch in range(NUM_EPOCH):
    logger.info('start {} epoch'.format(epoch))
    train_acc, train_loss, train_kts, train_spearman = [], [], [], []
    tatkc.train()
    MLP_model.train()
    tatkc.to(device)
    MLP_model.to(device)

    for j in tqdm(range(len(train_real_ts_l))):
        tatkc.ngh_finder = train_real_ngh_finder[j]
        m_loss = []

        TRAIN_BATCH_SIZE = BATCH_SIZE
        num_train_instance = len(nodeList_train_real[j])
        num_train_batch = math.ceil(num_train_instance / TRAIN_BATCH_SIZE)
        for k in range(num_train_batch):
            s_idx = k * TRAIN_BATCH_SIZE
            e_idx = min(num_train_instance, s_idx + TRAIN_BATCH_SIZE)
            src_l_cut = np.array(nodeList_train_real[j][s_idx:e_idx])
            label_l_cut = train_label_l_real[j][s_idx:e_idx]
            ts_l_cut = train_real_ts_list[j][s_idx:e_idx]
            optimizer.zero_grad()
            scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)
            src_embed = tatkc.tem_conv(src_l_cut, ts_l_cut, NUM_LAYER, num_neighbors=NUM_NEIGHBORS)
            true_label = torch.from_numpy(np.array(label_l_cut)).float().to(device)
            pred_bc = MLP_model(src_embed)

            loss = loss_cal(pred_bc, true_label, len(pred_bc), device)  # ranking loss
            m_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(list(tatkc.parameters()) + list(MLP_model.parameters()), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss.append(np.mean(m_loss))

    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(train_loss)))

train_end = time.time()
print('train end, train_time={}'.format(train_end - start_train))


real_data_start_time = time.time()
test_real_acc, test_real_kts = eval_real_data('test for real data', tatkc, MLP_model, test_real_ngh_finder,
                                              nodeList_test_real, test_real_ts_list, test_label_l_real)
real_data_end_time = time.time()
logger.info('Test real data statistics: nodes -- Top_1%: {}, Top_5%: {}, Top_10%: {}, Top_20%: {},kt:{}, time:{}'
            .format(test_real_acc[0], test_real_acc[1], test_real_acc[2], test_real_acc[3], test_real_kts,
                    real_data_end_time - real_data_start_time))
# torch.save(MLP_model.state_dict(), './saved_models/model_MLP.pth')
# print("MLP_model save over")
# torch.save(tatkc.state_dict(), './saved_models/model_tatkc.pth')
# print("tatkc_model save over")
