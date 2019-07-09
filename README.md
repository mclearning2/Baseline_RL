# Baseline RL algorithms for OpenAI gym


OpenAI gym의 환경들을 이용해서 [Spinning up의 Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)에 나오는 논문들을 최대한 많이 구현해보는 것이 목표인 프로젝트입니다. 또한 최대한 한 알고리즘이 다양한 환경에 적용될 수 있도록 실험자를 위한 프레임워크를 구축하는 것 또한 목표입니다.

또한 OpenAI의 multiprocessing_env를 활용하여 학습하고 여러 개 렌더링을 동시에 할 수 있습니다.(`n_worker` 파라미터 활용)

| BipedalWalker-v2 | CartPole-v0 | Pendulum-v0 |
| ---------------- | ----------- | ----------- |
| ![BipedalWalker-v2](https://github.com/mclearning2/Baseline_RL/blob/master/images/BipedalWalker-v2.gif) | ![CartPole-v0](https://github.com/mclearning2/Baseline_RL/blob/master/images/CartPole-v0.gif) | ![Pendulum-v0](https://github.com/mclearning2/Baseline_RL/blob/master/images/Pendulum-v0.gif) |

| Acrobot-v1 | LunarLander-v2 |
| ---------- | -------------- |
| ![Acrobot-v1](https://github.com/mclearning2/Baseline_RL/blob/master/images/Acrobot-v1.gif) | ![LunarLander-v2](https://github.com/mclearning2/Baseline_RL/blob/master/images/LunarLander-v2.gif) |

## Getting started

### Prerequisites

저의 실험환경. 그 외에는 돌아갈지 알 수 없음..

- Ubuntu 18.04 : 실제로 저는 linux mint 19.1에서 했지만 우분투에서도 잘 돌아감을 확인했습니다.
- Python 3.6.7

#### Package

```
sudo apt-get install python3-setuptools python-dev python3-dev python3-opencv swig ffmpeg build-essential libssl-dev libffi-dev libxml2-dev libxslt-dev libxslt1-dev zlib1g-dev qtbase5-dev libqt5opengl5-dev libassimp-dev cmake patchelf
python3 -m pip install virtualenv 
```

#### Repository

```
git clone https://github.com/mclearning2/Baseline_RL.git
```

#### Python Libraries

```
python3 -m virtualenv .env --no-site-packages
source .env/bin/activate
python3 -m pip install -r requirements.txt
```

#### Wandb Init

Please refer to https://docs.wandb.com/docs/started.html

```
wandb init
wandb login
```

#### Roboschool

```
sudo apt-get install qtbase5 libqt5opengl5-dev libassimp-dev patchelf cmake
git clone https://github.com/openai/roboschool && cd roboschool
./install_boost.sh
./install_bullet.sh
source exports.sh
export CPLUS_INCLUDE_PATH=/usr/include/python3.6
pip3 install wheel
cd roboschool/cpp-household && make clean && make -j4 && cd ../.. && pip3 install -e .
```

#### Train

``` bash
python3 main.py
```

![train.gif](https://github.com/mclearning2/Baseline_RL/blob/master/images/Train.gif)

#### Test

``` bash

python3 main.py --test
```

![train.gif](https://github.com/mclearning2/Baseline_RL/blob/master/images/Test.gif)

### Argument Parsing

| Argument | Default |Description |
| :--------: |:------: |:-------- |
| -\-project  | None | Restore from wandb by this project. and if None, this is replaced to the file name in {projects} folder you selected |
| -\-seed | 1 | Seed for reproduction |
| -\-test | - | If this is included, agent starts test not train |
| -\-restore | - | If this is included, hyperparameters from `./{report_dir}/model/{project}/hyperparams.pkl` is loaded and replaced by this in `hyperparams`  |
| -\-render | - | If this is included, agent starts render |
| -\-record | - | If this is included, Interaction is recorded in `./{report_dir}/videos/{project}/` |

## Environment

### Classic control

- [x] CartPole-v0
- [x] Pendulum-v0
- [ ] MountainCar-v0
- [ ] MountainCarContinuous-v0
- [x] Acrobot-v1

### Box2D

- [x] BipedalWalker-v2
- [ ] BipedalWalkerHardcore-v2
- [x] LunarLander-v2
- [ ] LunarLanderContinuous-v2
- [ ] CarRacing-v0

### Atari, MuJoCo, Roboschool

Comming soon...

## Algorithms

### Model-Free RL
- [x] **(DQN)** Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013.
- [ ] **(Deep Recurrent Q-Learning)** Deep Recurrent Q-Learning for Partially Observable MDPs, Hausknecht and Stone, 2015. 
- [x] **(Dueling DQN)** Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2015. 
- [x] **(Double DQN)** Deep Reinforcement Learning with Double Q-learning, Hasselt et al 2015.
- [x] **(PER)** Prioritized Experience Replay, Schaul et al, 2015.
- [ ] **(Rainbow DQN)** Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al, 2017.

### Policy Gradients
- [ ] **(REINFORCE)** Policy Gradient Methods for Reinforcement Learning with Function Approximation Richard et al
- [x] **(A2C)**
- [ ] **(A3C)** Asynchronous Methods for Deep Reinforcement Learning, Mnih et al, 2016.
- [ ] Trust Region Policy Optimization, Schulman et al, 2015. Algorithm: TRPO.
- [x] **(GAE)** High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al, 2015.
- [x] **(PPO-Clip)** Proximal Policy Optimization Algorithms, Schulman et al, 2017.
- [ ] **(PPO-Penalty)** Emergence of Locomotion Behaviours in Rich Environments, Heess et al, 2017.
- [ ] **(ACKTR)** Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation, Wu et al, 2017.
- [ ] **(ACER)** Sample Efficient Actor-Critic with Experience Replay, Wang et al, 2016.
- [ ] **(SAC)** Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018.

### Deterministic Policy Gradients

- ~~**(DPG)** Deterministic Policy Gradient Algorithms, Silver et al, 2014.~~
- [x] **(DDPG)** Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015.
- [x] **(TD3)** Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018.