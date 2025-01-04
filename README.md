# DualOptim


## Installation

Please check below requirements and install packages from `requirements.txt`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```


## Usage

The following command is an `symmetric 0.0-0.4` and `mixed 0.0-0.4` example of running the code.

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
#
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

```bash
```


### Parameters for noisy label
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `noise_type_lst` |  Noisy type list. |
| `noisy_group_num`  | Number of clients corresponding to noisy type. |
| `group_noise_rate` | The noise rate corresponding to the noisy group. It increases linearly from 0.0 to noise rate for each group. |


Please check `run.sh` for commands for various data and noisy label scenarios.
