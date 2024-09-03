# Policy Regularization with Dataset Constraint for Offline Reinforcement Learning

Code for ICML'23 paper "Policy Regularization with Dataset Constraint for Offline Reinforcement Learning", [arXiv link](https://arxiv.org/abs/2306.06569).

## Install dependency

```bash
pip install -r requirements.txt
```

Install the [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd d4rl
pip install -e .
```

## Run experiment

For halfcheetah:

```bash
python main.py --env_id halfcheetah-medium-v2 --seed 1024 --device cuda:0 --alpha 40.0 --beta 2.0 --k 1
```

For hopper & walker2d:

```bash
python main.py --env_id hopper-medium-v2 --seed 1024 --device cuda:0 --alpha 2.5 --beta 2.0 --k 1
```

We use reward shaping for antmaze, which is a common trick used by CQL, IQL, FisherBRC, etc.

```bash
python main.py --env_id antmaze-medium-play-v2 --seed 1024 --device cuda:0 --alpha 7.5 --beta 7.5 --k 1 --scale=10000 --shift=-1
```

## See result

```bash
tensorboard --logdir='./result'
```

## fixing install errors

### d4rl errors

The following commands are useful when import d4rl

```bash
pip install "cython<3"
apt-get install libosmesa6-dev
apt-get install patchelf
```

if cannot make mujoco envs try:

```bash
pip install six
```

because the d4rl do not register it due to sone error

### gym errors

The gym assert to be 0.21.0, so following command will replace the pip version

```bash
pip install wheel==0.38.4 setuptools==66.0.0
pip install "pip<24.1"
```