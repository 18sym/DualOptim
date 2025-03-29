# DualOptim

## Introduction
Official codes for our paper: Refining Distributed Noisy Clients: An End-to-end Dual Optimization Framework. This paper is now preprinted on Techrxiv via [this link](https://www.techrxiv.org/users/691169/articles/1258369-refining-distributed-noisy-clients-an-end-to-end-dual-optimization-framework).

## Installation

Please check below requirements and install packages from `requirements.txt`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## Usage

The following commands are examples representing `symmetric 0.0-0.4`, `pairflip 0.0-0.4` and `mixed 0.0-0.4` label noise settings.

```bash
# symmetric 0.0-0.4 dirichlet 1.0 

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 0.5 \
--method dualoptim | tee  symmetric04_dir05.txt

```

```bash
# pairflip 0.0-0.4 dirichlet 1.0 

python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst pairflip \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--partition dirichlet \
--dd_alpha 0.5 \
--method dualoptim | tee  pairflip04_dir05.txt

```


```bash
# mixed 0.0-0.4
python main_fed_LNL.py \
--dataset cifar10 \
--model resnet18 \
--epochs 120 \
--noise_type_lst symmetric pairflip\
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 0.0 0.4 \
--partition dirichlet \
--dd_alpha 1.0 \
--method dualoptim | tee mixed04_cifar10_dir1.txt
```



### Parameters for noisy label
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `noise_type_lst` |  Noisy type list. |
| `noisy_group_num`  | Number of clients corresponding to noisy type. |
| `group_noise_rate` | The noise rate corresponding to the noisy group. It increases linearly from 0.0 to noise rate for each group. |


Please check `run.sh` for commands for various data and noisy label scenarios.


# Citing this work
If you find this work helpful, please consider crediting our work:
```
 @article{DualOptim,
title={Refining Distributed Noisy Clients: An End-to-end Dual Optimization Framework},
url={http://dx.doi.org/10.36227/techrxiv.173707406.66001019/v1},
DOI={10.36227/techrxiv.173707406.66001019/v1},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Jiang, Xuefeng and Li, Peng and Sun, Sheng and Li, Jia and Wu, Lvhua and Wang, Yuwei and Lu, Xiuhua and Ma, Xu and Liu, Min},
year={2025},
month=jan
}
```
