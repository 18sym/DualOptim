#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
from sklearn.mixture import GaussianMixture

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast as autocast
import copy


from .correctors import SelfieCorrector, JointOptimCorrector
# from .nets import get_model
from resnets.build_model import build_model as get_model

import torchvision
from torchvision import transforms


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, real_idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.real_idx_return = real_idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        if self.idx_return:
            return image, label, item
        elif self.real_idx_return:
            return image, label, item, self.idxs[item]
        else:
            return image, label


class PairProbDataset(Dataset):
    def __init__(self, dataset, idxs, prob, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.prob = prob

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        prob = self.prob[self.idxs[item]]

        if self.idx_return:
            return image1, image2, label, prob, item
        else:
            return image1, image2, label, prob


class PairDataset(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, label_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.label_return = label_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        sample = (image1, image2,)

        if self.label_return:
            sample += (label,)

        if self.idx_return:
            sample += (item,)

        return sample


class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        return image, label, self.idxs[item]


def mixup(inputs, targets, alpha=1.0):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    idx = torch.randperm(inputs.size(0))

    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target


def linear_rampup(current, warm_up, lambda_u, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class SemiLoss:
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        # labeled data loss
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x,
                         dim=1) * targets_x, dim=1))
        # unlabeled data loss
        Lu = torch.mean((probs_u - targets_u) ** 2)

        lamb = linear_rampup(epoch, warm_up, lambda_u)

        return Lx + lamb * Lu


def get_local_update_objects(args, dataset_train, dict_users=None, noise_rates=None, gaussian_noise=None, glob_centroid=None):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )

        # if args.method == 'robustfl':
        #     local_update_args_rfl = dict(
        #         args=args,
        #         user_idx=idx,
        #         dataset=dataset_train,
        #         idxs=dict_users[idx],
        #         glob_centroid=glob_centroid
        #     )

        if args.method == 'dualoptim':
            local_update_object = LocalUpdateDualOptim(**local_update_args)
        else:
            raise NotImplementedError

        local_update_objects.append(local_update_object)

    return local_update_objects


class SAM(torch.optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.9, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2)
                       if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get back to "w" from "w + e(w)"
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0)
                         * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

#TODO: 
class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8
        

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class BaseLocalUpdate:
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = args.method

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.total_epochs = 0
        self.epoch = 0
        self.batch_idx = 0

        self.net1 = get_model(self.args)
        self.net2 = get_model(self.args)
        # self.net1 = self.net1.to(self.args.device)
        # self.net2 = self.net2.to(self.args.device)

        self.last_updated = 0

        self.is_svd_loss = args.is_svd_loss
        self.feddecorr = FedDecorrLoss()
        self.decorr_coef = args.decorr_coef

    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        # net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx

                if(len(batch) == 0):
                    continue

                net.zero_grad()

                # with autocast():
                loss = self.forward_pass(batch, net)

                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss.item():.6f}")

                batch_loss.append(loss.item())
                self.on_batch_end()

            if(len(batch_loss) > 0):
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_multiple_models(self, net1, net2):

        # net1.to(self.args.device)
        # net2.to(self.args.device)

        net1.train()
        net2.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer1 = torch.optim.SGD(net1.parameters(), **optimizer_args)
        optimizer2 = torch.optim.SGD(net2.parameters(), **optimizer_args)

        epoch_loss1 = []
        epoch_loss2 = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net1.zero_grad()
                net2.zero_grad()

                # with autocast():
                loss1, loss2 = self.forward_pass(batch, net1, net2)

                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss1.item():.6f}"
                          f"\tLoss: {loss2.item():.6f}")

                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
                self.on_batch_end()

            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net1.state_dict())
        self.net2.load_state_dict(net2.state_dict())
        self.last_updated = self.args.g_epoch

        # net1.to('cpu')
        # net2.to('cpu')
        # del net1
        # del net2

        return self.net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), \
            self.net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)

    def forward_pass(self, batch, net, net2=None):
        if self.idx_return:
            images, labels, _ = batch

        elif self.real_idx_return:
            images, labels, _, ids = batch
        else:
            images, labels = batch

        images = images.to(self.args.device)
        labels = labels.to(self.args.device)

        with autocast():
            log_probs, features = net(images)
        loss = self.loss_func(log_probs, labels)
        if self.is_svd_loss:
            loss_feddecorr = self.feddecorr(features)
            loss += loss_feddecorr * self.decorr_coef

        if net2 is None:
            return loss

        # 2 models
        with autocast():
            log_probs2, features2 = net2(images)
        loss2 = self.loss_func(log_probs2, labels)

        if self.is_svd_loss:
            loss_feddecorr2 = self.feddecorr(features2)
            loss2 += loss_feddecorr2 * self.decorr_coef

        return loss, loss2

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass

class LocalUpdateDualOptim(BaseLocalUpdate):
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=True,
    ):
        self.args = args

        self.dataset = dataset
        self.idxs = idxs

        self.user_idx = user_idx
        self.update_name = 'dualoptim'

        self.idx_return = idx_return
        self.real_idx_return = True
        self.total_epochs = 0


        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.ldr_train_infer = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        self.ldr_train_tmp = DataLoader(DatasetSplitRFL(
            dataset, idxs), batch_size=1, shuffle=True)

        self.class_sum = np.array([0] * args.num_classes)
        for idx in self.idxs:
            label = self.dataset.train_labels[idx]
            self.class_sum[label] += 1

        from utils.losses import LogitAdjust, LA_KD, LogitAdjust_soft
        self.loss_func1 = LogitAdjust(cls_num_list=self.class_sum)
        self.loss_func_soft = LogitAdjust_soft(cls_num_list=self.class_sum)
        self.loss_func2 = LA_KD(cls_num_list=self.class_sum)
        self.loss_func = torch.nn.CrossEntropyLoss().to(args.device)
        self.net1 = get_model(self.args)
        self.last_updated = 0

        self.is_svd_loss = args.is_svd_loss
        self.feddecorr = FedDecorrLoss()
        self.decorr_coef = args.decorr_coef
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.epoch_of_stage1 = args.epoch_of_stage1

        self.local_datasize = len(idxs)

        self.index_mapper, self.index_mapper_inv = {}, {}

        for i in range(len(self.idxs)):
            self.index_mapper[self.idxs[i]] = i
            self.index_mapper_inv[i] = self.idxs[i]

        self.label_update = torch.index_select(
            args.Soft_labels, 0, torch.tensor(self.idxs))
        # yy = torch.FloatTensor(yy)
        self.label_update = torch.FloatTensor(self.label_update)

        self.true_labels_local = torch.index_select(
            args.True_Labels, 0, torch.tensor(self.idxs))

        self.estimated_labels = copy.deepcopy(self.label_update)

        # yield by the local model after E local epochs
        self.final_prediction_labels = copy.deepcopy(self.label_update)

        # self.estimated_labels = F.softmax(self.label_update, dim=1)
        self.lamda = args.lamda_pencil

        for batch_idx, batch in enumerate(self.ldr_train_infer):

            if self.idx_return:
                images, labels, _ = batch
            elif self.real_idx_return:
                images, labels, _, ids = batch
            else:
                images, labels = batch

            indexss = self.indexMapping(ids)
            # self.label_update[indexss].cuda()
            for i in range(len(indexss)):
                self.label_update[indexss[i]][labels[i]] = 10  # self.args.K_pencil

        print(f"Initializing the client #{self.user_idx}... Done")

    # from overall index to local index
    def indexMapping(self, indexs):
        indexss = indexs.cpu().numpy().tolist()
        target_mapping = []
        for each in indexss:
            target_mapping.append(self.index_mapper[each])
        return target_mapping

    def label_updating(self, labels_grad):
        self.label_update = self.label_update - self.lamda * labels_grad
        # if self.lamda > 50:
        #     self.lamda = self.lamda - 50
        self.estimated_labels = F.softmax(self.label_update, dim=1)

    def pencil_loss(self, outputs, labels_update, labels, feat, feature_centers, max_gap):

        pred = F.softmax(outputs, dim=1)
        # yd = F.softmax(labels_update, dim=1)

        features_square = torch.sum(torch.pow(feat, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(feature_centers, 2), 1, keepdim=True)

        features_into_centers = torch.matmul(feat, feature_centers.T)
        dist_2 = features_square - 2 * features_into_centers + centers_square.T
        dist_2 = torch.sqrt(dist_2)
        one_hot = F.one_hot(labels, self.args.num_classes).to(self.args.device)

        gap = min(max_gap.item(), 100)
        dist_2 = dist_2 + one_hot * gap

        center_loss = self.loss_func(-dist_2, labels)

        Lc = -torch.mean(torch.sum(F.log_softmax(labels_update, dim=1) * pred, dim=1))
        Lo = -torch.mean(F.log_softmax(labels_update, dim=1)[torch.arange(labels_update.shape[0]), labels])

        loss_total = (Lc / self.args.num_classes) + (Lo * 0.6) + (center_loss * 0.4)
        print(f"lc: {Lc},  lo:{Lo}, center_loss: {center_loss}:, in total，lc:{(Lc / self.args.num_classes)}, lo:{(Lo * 0.6)}, center_loss:{(center_loss * 0.4)}")


        return loss_total

    def get_small_loss_samples(self, y_pred, y_true, forget_rate):
        loss = self.loss_func(y_pred, y_true)

        ind_sorted = np.argsort(loss.data.cpu()).to(self.args.device)

        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate

        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        return ind_update

    def train_stage1(self, net):  # train with LA
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []


        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []

            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch
                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                logits, feat = net(images)

                loss = self.loss_func1(logits, labels)

                if self.is_svd_loss:
                    loss_feddecorr = self.feddecorr(feat)
                    loss += loss_feddecorr * self.decorr_coef

                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss.item():.6f}")

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            self.total_epochs += 1

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_stage2(self, net, global_net, weight_kd, current_round, f_G, min_gapp, max_gapp):
        feature_centers = torch.zeros((self.args.num_classes, self.args.feature_dim), device=self.args.device)
        class_counts = torch.zeros(self.args.num_classes, device=self.args.device)
        epoch = current_round

        if epoch == self.epoch_of_stage1:

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.ldr_train):
                    self.batch_idx = batch_idx
                    net.zero_grad()

                    if self.idx_return:
                        images, labels, _ = batch

                    elif self.real_idx_return:
                        images, labels, _, ids = batch
                    else:
                        images, labels = batch

                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)

                    logits, feat = net(images)
                    feat = feat.detach()
                    for i in range(feat.size(0)):
                        feature_centers[labels[i]] += feat[i]
                        class_counts[labels[i]] += 1

                valid_classes = class_counts != 0
                feature_centers[valid_classes] /= class_counts[valid_classes].unsqueeze(1)
                feature_centers[~valid_classes] = torch.randn(1, self.args.feature_dim, device=self.args.device)
                feature_centers = feature_centers.half()

                gap = torch.ones((self.args.num_classes, self.args.num_classes), device=self.args.device) * 1e9
                for i in range(self.args.num_classes):
                    for j in range(i):
                        dis = torch.norm(feature_centers[i] - feature_centers[j], p=2)
                        gap[i, j] = dis
                        gap[j, i] = dis
                min_gap = torch.min(gap[torch.triu(torch.ones_like(gap), diagonal=1) > 0])
                max_gap = torch.max(gap[torch.triu(torch.ones_like(gap), diagonal=1) > 0])

        else:
            feature_centers = f_G.half()
            min_gap = min_gapp
            max_gap = max_gapp

        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []

        before_correct_predictions = 0
        for batch_idx, batch in enumerate(self.ldr_train_infer):

            if self.idx_return:
                images, labels, _ = batch
            elif self.real_idx_return:
                images, labels, _, ids = batch
            else:
                images, labels = batch

            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            local_index = self.indexMapping(ids)

            with torch.no_grad():
                teacher_output, teacher_feat = global_net(images)
                soft_label = torch.softmax(teacher_output, dim=1)
                soft_label = soft_label.to('cpu')
                predicted_classes = torch.argmax(soft_label, dim=1).to('cpu')
                correct_predictions = (predicted_classes == self.true_labels_local[local_index]).sum().item()

                before_correct_predictions += correct_predictions

        before_percentage_correct = (before_correct_predictions / len(self.idxs)) * 100
        print(
            f'########[local prediction before training]######### For client#{self.user_idx}, global model correctly predicts {before_percentage_correct}% local samples...')

        # # from overall index (ids) to local index (indexss)
        # indexss = self.indexMapping(ids)
        # # self.label_update[indexss].cuda()
        # for i in range(len(indexss)):
        #     self.label_update[indexss[i]] = soft_label[i] * self.args.K_pencil
        #     # if self.user_idx == 99:
        #     # print(f"for {i} iter, local index {indexss[i]}, label is {labels[i]}, real label is {self.true_labels_local[indexss[i]]}")

        # print(f"PREVIOUS label update distribution is: {self.label_update[0]}")


        # TODO: To begin the local training
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []


            labels_grad = torch.zeros(self.local_datasize, self.args.num_classes, dtype=torch.float32)
            # labels_grad = np.zeros(
            # (self.local_datasize, self.args.num_classes), dtype=np.float32)
            # labels_grad = torch.tensor(labels_grad)

            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch

                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch

                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                with autocast():
                    logits, feat = net(images)
                    feat_copy = feat.clone()
                    feat = feat.detach()

                indexss = self.indexMapping(ids)
                # print(f"^^^^^^^^^[IDs]: {ids[:10]}")
                # print(f"^^^^^^^^^[indexss]: {indexss[:10]}")


                labels_update = self.label_update[indexss, :].cuda()
                labels_update.requires_grad_()
                # labels_update = torch.autograd.Variable(labels_update,requires_grad = True)

                loss = self.pencil_loss(
                    logits, labels_update, labels, feat, feature_centers, max_gap)

                if self.is_svd_loss:
                    loss_feddecorr = self.feddecorr(feat_copy)
                    loss += loss_feddecorr * self.decorr_coef

                loss.backward()

                labels_grad[indexss] = labels_update.grad.cpu().detach()  # .numpy()

                labels_update = labels_update.to('cpu')
                del labels_update

                optimizer.step()

                # if self.args.verbose and batch_idx % 10 == 0:
                #     print(f"Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                #           f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss.item():.6f}")

                batch_loss.append(loss.item())

            self.label_updating(labels_grad)


            # print(f"&&&local epoch#{epoch}, the label grad is: {labels_grad[0]}")
            # print(f"&&&local epoch#{epoch}, the label update distribution is: {self.label_update[0]}")
            # print(f"&&&local epoch#{epoch}, the estimated label is: {self.estimated_labels[0]}")
            # print(f"&&&local epoch#{epoch}, the GT label is: {self.true_labels_local[0]}") #self.index_mapper_inv[i]

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1



            with torch.no_grad():
                for batch_idx, batch in enumerate(self.ldr_train):
                    self.batch_idx = batch_idx
                    net.zero_grad()

                    if self.idx_return:
                        images, labels, _ = batch

                    # TODO: we use the below one
                    elif self.real_idx_return:
                        images, labels, _, ids = batch
                    else:
                        images, labels = batch

                    images = images.to(self.args.device)
                    labels = labels.to(self.args.device)

                    logits, feat = net(images)
                    feat = feat.detach()
                    for i in range(feat.size(0)):
                        feature_centers[labels[i]] += feat[i]
                        class_counts[labels[i]] += 1


                valid_classes = class_counts != 0
                feature_centers[valid_classes] /= class_counts[valid_classes].unsqueeze(1)
                feature_centers[~valid_classes] = torch.randn(1, self.args.feature_dim, device=self.args.device).half()
                feature_centers = feature_centers.half()


        # After E local epochs
        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        # print(f"SCALE TEST: estimated {self.estimated_labels.shape} ; true {self.true_labels_local.shape}")

        predicted_classes = torch.argmax(self.estimated_labels, dim=1)
        correct_predictions = (predicted_classes == self.true_labels_local).sum().item()
        total_samples = self.true_labels_local.size(0)
        percentage_correct = (correct_predictions / total_samples) * 100

        print(
            f'########[local estimate]######### For client#{self.user_idx}, we correctly estimate {percentage_correct:.2f}% local samples...')

        # self.estimated_labels = F.softmax(self.label_update, dim=1)
        # self.true_labels_local = torch.index_select(
        #     self.args.True_Labels, 0, torch.tensor(self.idxs))
        # SCALE TEST: estimated torch.Size([547, 10]) ; true torch.Size([547])

        after_correct_predictions = 0
        for batch_idx, batch in enumerate(self.ldr_train_infer):

            if self.idx_return:
                images, labels, _ = batch
            elif self.real_idx_return:
                images, labels, _, ids = batch
            else:
                images, labels = batch

            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            local_index = self.indexMapping(ids)

            # with autocast():
            with torch.no_grad():
                output_final, teacher_feat = net(images)
                output_final = output_final.to('cpu')

                soft_label = torch.softmax(output_final, dim=1)
                self.final_prediction_labels[local_index] = soft_label

                predicted_classes = torch.argmax(soft_label, dim=1)
                correct_predictions = (predicted_classes == self.true_labels_local[local_index]).sum().item()

                after_correct_predictions += correct_predictions

        net.to('cpu')
        del net

        after_percentage_correct = (after_correct_predictions / len(self.idxs)) * 100
        print(
            f'########[local prediction after training]######### For client#{self.user_idx}, after training, global model correctly predicts {after_percentage_correct}% local samples...')

        # TODO: merge the softmax(self.label_update) and the prediction after local training(self.final_prediction_labels)
        self.label_update = self.label_update.to('cpu')

        updated_local_labels_tmp = F.softmax(self.label_update, dim=1)
        final_model_prediction_tmp = self.final_prediction_labels
        # average the above two
        merged_local_labels = (updated_local_labels_tmp + final_model_prediction_tmp) / 2
        # the GT labels is self.true_labels_local
        predicted_classes = torch.argmax(merged_local_labels, dim=1)
        merged_correct_predictions = (predicted_classes == self.true_labels_local).sum().item()
        merged_total_samples = self.true_labels_local.size(0)
        merged_percentage_correct = (merged_correct_predictions / merged_total_samples) * 100
        print(
            f'########[local prediction after merging]######### For client#{self.user_idx}, after merging, global model correctly predicts {merged_percentage_correct}% local samples...')

        # replace the label_update with the merged_local_labels, and rescale by K_pencil
        self.label_update = merged_local_labels * self.args.K_pencil

        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(
            epoch_loss), percentage_correct, before_percentage_correct, after_percentage_correct, merged_percentage_correct,feature_centers
