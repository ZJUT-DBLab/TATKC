import json
import pickle

import numpy as np
import pandas as pd
import torch

from graph import NeighborFinder


def preprocess(data_name):
    u_list, i_list, ts_list = [], [], []
    idx_list = []

    with open(data_name) as f:
        # s = next(f)
        # print(s)
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            # values = [v.split(' ') for v in e]
            u = int(e[0])
            i = int(e[1])
            ts_float = float(e[2])
            ts = int(ts_float)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            idx_list.append(idx)

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list})


def run(data_name):

    PATH = './data/test/Real/{}.txt'.format(data_name)
    OUT_DF = './data/test/Real/processed/ml_{}.csv'.format(data_name)
    OUT_NODE_FEAT = './data/test/Real/processed/ml_{}_node.npy'.format(data_name)

    df = preprocess(PATH)

    num_total_unique_nodes = max(df.u.values.max(), df.i.values.max())

    feat = 128
    rand_feat = np.zeros((num_total_unique_nodes + 1, feat))

    df.to_csv(OUT_DF)
    np.save(OUT_NODE_FEAT, rand_feat)

# run('edit-tgwiktionary')
# run('edit-mgwikipedia')
# run('edit-ltwiktionary')
# run('edit-mlwikiquote')
# run('edit-plwikiquote')
# run('edit-warwikipedia')
# run('edit-zhwiktionary')
# run('edit-mgwiktioanry')
