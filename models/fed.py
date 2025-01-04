#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
# from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# TODO: This file manages all the clients' state and the aggregation protocals


class LocalModelWeights:
    def __init__(self, all_clients, net_glob, num_users, method, dict_users, args):
        self.all_clients = all_clients
        self.num_users = num_users
        self.method = method
        self.args = args
        self.user_data_size = [len(dict_users[i]) for i in range(num_users)]
        
        # if the data size for each user is the same,
        # set all elements in user_data_size as 1
        if self.user_data_size and \
                all([self.user_data_size[0] == data_size for data_size in self.user_data_size]):
            self.user_data_size = [1] * len(self.user_data_size)

        self.model_ = copy.deepcopy(net_glob)
        w_glob = net_glob.state_dict()
        self.global_w_init = net_glob.state_dict()  # which can be used for FedExp

        if self.all_clients:
            print("Aggregation over all clients")
            self.w_locals = [w_glob for i in range(self.num_users)]
            self.data_size_locals = self.user_data_size
        else:
            self.w_locals = []
            self.data_size_locals = []

    def init(self):
        # Reset local weights if necessary
        if not self.all_clients:
            self.w_locals = []
            self.data_size_locals = []

    def update(self, idx, w):
        if self.all_clients:
            self.w_locals[idx] = copy.deepcopy(w)
        else:
            self.w_locals.append(copy.deepcopy(w))
            self.data_size_locals.append(self.user_data_size[idx])

    def average(self):
        w_glob = None
        # approaches for original methods which not modify the aggregation process
        if self.method == 'dualoptim':
            if self.noisy_clients == 0:
                w_glob = FedAvg(self.w_locals, self.data_size_locals)
            else:
                #FIXME: 10 is computed by 100*client_selection_ratio(0.1), we can modify it to make it more flexible
                assert len(self.client_tag) == self.args.selected_total_clients_num, self.client_tag  # 这里的client_tag在每一轮的最后都重新设置为[]

                if len(set(self.client_tag)) == 1: # all clean or all noisy, use FedAvg
                    w_glob = FedAvg(self.w_locals, self.data_size_locals)
                else:
                    w_glob = DaAgg(
                        self.w_locals, self.data_size_locals, self.client_tag)

                self.client_tag = [] # initial again for the next round
        else:
            # default method for aggregation
            w_glob = FedAvg(self.w_locals, self.data_size_locals)
            # exit('Error: unrecognized aggregation method')

        return w_glob


# average_weights is a list of weights for each client
def FedAvg(w, average_weights):
    global_w_update = copy.deepcopy(w[0])
    for k in global_w_update.keys():
        global_w_update[k] *= average_weights[0]
        for i in range(1, len(w)):
            global_w_update[k] += w[i][k] * average_weights[i]
        global_w_update[k] = torch.div(
            global_w_update[k], sum(average_weights))

    return global_w_update


def DaAgg(w, dict_len, client_tag):
    client_weight = np.array(dict_len)
    client_weight = client_weight / client_weight.sum()

    clean_clients = []
    noisy_clients = []
    for index, element in enumerate(client_tag):
        if element == 1:
            clean_clients.append(index)
        elif element == 0:
            noisy_clients.append(index)
        else:
            raise
    
    distance = np.zeros(len(dict_len))
    for n_idx in noisy_clients:
        dis = []
        for c_idx in clean_clients:
            dis.append(model_dist(w[n_idx], w[c_idx]))
        distance[n_idx] = min(dis)
    distance = distance / distance.max()
    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / client_weight.sum()
    # print(client_weight)

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * client_weight[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * client_weight[i]
    return w_avg


def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1.keys():
        if "int" in str(w_1[key].dtype):
            continue
        dist = torch.norm(w_1[key] - w_2[key])
        dist_total += dist.cpu()

    return dist_total.cpu().item()

  
def Median(w):  
    global_w_update = copy.deepcopy(w[0])  
    num_models = len(w)  
  
    for k in global_w_update.keys():  
        parameter_values = [w[i][k] for i in range(num_models)]  
        aggregated_parameter = torch.median(torch.stack(parameter_values, dim=0), dim=0).values  
  
        global_w_update[k] = aggregated_parameter  
  
    return global_w_update


def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)


def multi_vectorization(w_locals, args):
    vectors = copy.deepcopy(w_locals)

    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1])
        vectors[i] = torch.cat(list(v.values()))

    return vectors


def Krum(w_locals, c, args):
    n = len(w_locals) - c

    distance = pairwise_distance(w_locals, args)
    sorted_idx = distance.sum(dim=0).argsort()[: n]

    chosen_idx = int(sorted_idx[0])

    return copy.deepcopy(w_locals[chosen_idx])


def pairwise_distance(w_locals, args):
    vectors = multi_vectorization(w_locals, args)
    distance = torch.zeros([len(vectors), len(vectors)]).to(args.device)

    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)

    return distance


def fedavgg(w_locals):
    w_avg = copy.deepcopy(w_locals[0])

    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))

    return w_avg




def trimmed_mean(w_locals, c, args):
    n = len(w_locals) - 2 * c

    distance = pairwise_distance(w_locals, args)

    distance = distance.sum(dim=1)
    med = distance.median()
    _, chosen = torch.sort(abs(distance - med))
    chosen = chosen[: n]
        
    return fedavgg([copy.deepcopy(w_locals[int(i)]) for i in chosen])
