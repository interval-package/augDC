# Policy Regularization with Dataset Constraint for Offline Reinforcement Learning



## PRDC

Code for Policy Regularization with Dataset Constraint for Offline Reinforcement Learning (https://arxiv.org/abs/2306.06569)

If you find this repository useful for your research, please cite:

```
@journal{ran2023policy,
      title={Policy Regularization with Dataset Constraint for Offline Reinforcement Learning}, 
      author={Yuhang Ran and Yi-Chen Li and Fuxiang Zhang and Zongzhang Zhang and Yang Yu},
      year={2023},
      journal={arxiv Preprint arxiv:2306.06569}
}
```



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





