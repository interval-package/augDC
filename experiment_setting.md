# Experiment Setting

## Experiment for testing the simulators

找几篇关于世界模型的一些论文，要预测未来发生事情这样子。

我们用几种方法，训练一个仿真器。这个仿真器是用离线数据训练出来的。

这里仿真器，使用了一个余弦loss的小技巧

我们要表明就是我们的仿真器是和真实数据是两个分布，这边就出一个两边生成数据的差异图。
主要是要注意就是，我们也要关注模型多步预测之后导致的累计误差。

之前PRDC使用KD Tree去找最近邻，现在我们提前给它加一个Anchor，然后用Anchor去加速我们找的这个过程
主要目的是加速，我们的训练过程。

## Experiment for our algorithm

对比一下方法：
- 一个不带模型的方法
- PRDC
- 仿真器数据加强
- 我们的方法

### Test algorithm convergence speed

对比我们方法的收敛速度。

### Test algorithm OOD

就是比对，我们的算法能够非常有效减少我们的OOD问题

### Test algorithm performance

就是比对，我们模型的训练效果跟之前方法的效果上的差异

# Algorithm details

## Dataset Regulation

现在我们已经拿到了增强的数据，那怎么运用上，是挺麻烦的一个事情。

原来PRDC是只在Policy IMP的时候给新的策略梯度上加一部分内容。

对于我们增强的数据，我们在这里对Critic的训练也可以添加上去。(可能还是不要，Critic的内容还是要保证一下完整性的)，或者说找最近邻。

（这里理论要补充一下证明，从TRPO那边做一点类似的吧）

为什么不可以使用仿真器做online learning，这个也是需要讨论的一个要点。

这里的证明，可以假设是给定的简单的两种分布数据。其间的差距是可以度量的。

- 证明，带噪音的仿真器（当噪音高到什么地步），或者是这个仿真器生成的数据与真实分布的数据差异到多少的时候，我们online learning的方法，将会对模型不佳
- 证明，带噪音数据，怎么被约束到了我们的真实分布上面去的（也就是我们augDC的核心证明）
    - 证Critic，在这个约束下，更能贴合
    - 证Actor or Policy，OOD性质满足


## 示教轨迹

这里offline PRDC 也是一个bpo的过程

仿真器roll out出的轨迹，和最近的轨迹都算一个grad，然后两个grad combine一下
（注意，这里是我们用仿真器的增广数据学习的时候使用的方法）

# Implementation details

梯度融合的内容就用bpo那边的东西，证明不一定要`numbu`动力学。
