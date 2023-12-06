import numpy as np
import sys
import pickle as pkl
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


# 加载数据
def load_data(dataset_str):
    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    objects = []
    for i in range(len(names)):
        # Data(5)代表窗口为5，Data(one_hop)代表窗口为3
        with open("Data(5)/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            objects.append(np.load(f, allow_pickle=True))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, allx_adj, allx_embed, ally = tuple(objects)
    # train_idx_ori = parse_index_file("data50/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []
    train_embed = []
    val_adj = []
    val_embed = []
    test_adj = []
    test_embed = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)

    for i in range(len(y), len(ally)):  # train_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        val_adj.append(adj)
        val_embed.append(embed)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)

    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)
    train_y = torch.from_numpy(np.array(y))
    val_y = torch.from_numpy(np.array(ally[len(y):len(ally)]))  # train_size])
    test_y = torch.from_numpy(np.array(ty))

    return train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y


# 将所有数据转化为torch
def tras(data):
    return torch.from_numpy(data)


# 对称归一化邻接矩阵
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding

    for i in tqdm(range(adj.shape[0])):
        adj_normalized = adj[i] # no self-loop
        pad = max_length - adj_normalized.shape[0] # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
        mask[i, :adj[i].shape[0], :] = 1.
        adj[i] = adj_normalized

    # 返回类型为tensor类型的数据
    return torch.from_numpy(np.array(list(adj))), torch.from_numpy(mask) # coo_to_tuple(sparse.COO(np.array(list(adj)))), mask


# 处理特征矩阵(padding处理)
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    max_length = max([len(f) for f in features])

    for i in tqdm(range(features.shape[0])):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0]  # padding for each epoch
        feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    # 返回类型为tensor的数据
    return torch.from_numpy(np.array(list(features)))


def load_dataset(data, labels, batch_size):
    data_set = TensorDataset(data, labels)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, )

    return data_loader


def preprocess_adj1(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding

    for i in tqdm(range(adj.shape[0])):
        adj_normalized = adj[i] # no self-loop
        pad = max_length - adj_normalized.shape[0] # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
        # mask[i, :adj[i].shape[0], :] = 1.
        adj[i] = adj_normalized
        mask[i, :, :] = torch.from_numpy(MaxIndependentSet(adj[i])).unsqueeze(-1)
    # 返回类型为tensor类型的数据
    return torch.from_numpy(np.array(list(adj))), torch.from_numpy(mask)


def KatzCentrality(adj):
    I = np.eye(adj.shape[0])
    alpha = 1 / 100
    beta = 1.
    result = np.linalg.inv(I - alpha * adj) * beta
    return np.argsort(result.sum(0))


def MaxIndependentSet(adj):
    real_adj = np.array(adj, copy=True)
    seq = KatzCentrality(adj)
    new_mask = np.zeros(adj.shape[0])
    if real_adj[0].tolist().count(1) == 1:
        new_mask[0] = 1.
    A = np.zeros_like(adj)
    j=seq.shape[0]-1
    while 1:
        if real_adj[seq[j]].tolist().count(1) > 1:
            new_mask[seq[j]] = 1.
        real_adj[seq[j], :] = 0.
        real_adj[:, seq[j]] = 0.
        j = j-1
        if np.all(np.equal(real_adj, A)):
            break
    return new_mask
