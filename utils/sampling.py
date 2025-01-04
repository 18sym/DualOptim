#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np

def sample_dirichlet(labels, num_clients, alpha, num_classes=10):
    min_size = 0
    K = num_classes
    N = labels.shape[0]
    # print("N = " + str(N))
    net_dataidx_map = {}
    min_require_size = 100

    while min_size < min_require_size:
        #print(min_size)
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.seed(k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) 
            np.random.shuffle(idx_k)
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        print('idx: {}, size: {}'.format(j, len(net_dataidx_map[j])))
    return net_dataidx_map


# def sample_dirichlet(N, num_users, alpha=1.0, num_classes=10):
#     # 
#     dirichlet_distribution = np.random.dirichlet([alpha] * num_users)

#     # 
#     samples_per_user = (N * dirichlet_distribution).astype(int)

#     # 
#     samples_per_user[-1] += N - np.sum(samples_per_user)

#     # initialize dict_users
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

#     # 然后，我们需要将每个类别的样本均匀地分配到每个client
#     num_samples_per_class = N // num_classes
#     for i in range(num_classes):
#         class_indices = np.random.permutation(np.arange(i * num_samples_per_class, (i + 1) * num_samples_per_class))
#         start_index = 0
#         for user in range(num_users):
#             end_index = start_index + samples_per_user[user] // num_classes
#             dict_users[user] = np.concatenate((dict_users[user], class_indices[start_index:end_index]), axis=0)
#             print('idx: {}, size: {}'.format(user, len(dict_users[user])))
#             start_index = end_index

#     return dict_users

# def sample_dirichlet(labels, num_users, alpha=1.0):
#     min_size = 0
#     min_require_size = 10
#     K = 10
#     num_users = num_users

#     N = len(labels)
#     dict_users = {}

#     y_train = []

#     for i in range(K):
#         y_train += [i] * int(N / K)
#     y_train = np.array(y_train)

#     while min_size < min_require_size:
#         idx_batch = [[] for _ in range(num_users)]
#         for k in range(K):
#             idx_k = np.where(y_train == k)[0]

#             np.random.shuffle(idx_k)
#             proportions = np.random.dirichlet(np.repeat(alpha, num_users))
#             proportions = np.array([p * (len(idx_j) < N / num_users)
#                                     for p, idx_j in zip(proportions, idx_batch)])
#             proportions = proportions / proportions.sum()
#             proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

#             idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
#             min_size = min([len(idx_j) for idx_j in idx_batch])
#     tot = 0
#     for j in range(num_users):
#         np.random.shuffle(idx_batch[j])
#         dict_users[j] = idx_batch[j]
#         print('idx: {}, size: {}'.format(j, len(dict_users[j])))
#         if j < 50:
#             tot += len(dict_users[j])
#     print('total: {}'.format(tot))
#     return dict_users
