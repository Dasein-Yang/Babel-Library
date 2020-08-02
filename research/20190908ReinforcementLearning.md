# 20190908 强化学习分类



> 本笔记主要记录近年来比较有代表性的强化学习方法。 —— 2019 09-08

强化学习方法大致可以从三个维度分类，model-based/model free，on-policy /off-policy，value based/policy based。

## Model based/ Free

- model-based：主要发展自最优控制领域。通常先通过高斯过程(GP)或贝叶斯网络(BN)等工具针对具体问题建立模型,然后再通过机器学习的方法或最优控制的方法,如模型预测控制(MPC)、线性二次调节器(LQR)、线性二次高斯(LQG)、迭代学习控制(ICL)等进行求解。核心问题是如何学到 MDP 模型，实际上这类方法建模比较难。
- model-free：更多地发展自机器学习领域,属于数据驱动的方法。通过大量采样,估计代理的状态、动作的值函数或回报函数,从而优化动作策略。一般做法是学习策略，而不精确的估计模型，以下 valuebased 和 policybased 蒙特卡洛和TD基本都是无模型的。

## On Policy / Off Policy

- on-policy : 方法的通用套路是，用A策略 sample policy 一边采样得到 action-state 对P(s’|s,a)，更新B策略 behaviour policy 的参数，A=B即为 on target policy。
- off-policy: 反之，AB则为off target policy

REINFORCE 和 Actor-Critic都是onpolicy的，off-policy又被称之为experience replay，不需要一边采样一边更新。

## Value based /Policy based

- value based: 基于Q值估算，核心问题是如何估算更准。

- - -greedy： ε= 0.9 时有90% 的情况我会按照 Q 表的最优值选择行为, 10% 的时间使用随机选行为.
    - TD learning: 动态规划的实现
    - DQN：value function approximation 方法，它的思路主要还是学Q值，只不过使用神经网络近似Q值，还引入了 epsilon 贪心策略，以一定概率用随机或者最大化Q值的动作采样，再从采样中选择 minibatch 的数据来估算梯度更新网络参数。在离散空间中性能不错。
    - TD3 也是 value function approximation 这一类方法，在此基础上加了很多DDPGtrick，光从性能上看非常好。

- policy based: 假定好一个 policy 模型（神经网络），输入状态参数，输出 action，再迭代优化学习模型参数，也就是说要以原参数跑完多个 episode 优化。

- - REINFORCE: monte carlo 法跑一次或多跑几次游戏来估计一个全轨迹估计策略梯度，用于更新策略

- value-policy based: 同时基于 value 和 policy 。

- - Actor critic 结合了两种，Actor 用策略梯度更新策略，Critic 用值函数之差更新值函数模型。

单单从策略优化的方面又有几种思路，一种是 policy Interation（通过值函数的更新迭代自我），一种是 policy gradient（使用估计回报期望来估算 policy gradient），一种是无需可导的，引入其他数学框架（如最大熵）。

## Policy gradient

上述基于策略的方法中比较特殊的一类是policy gradient理论为基础的一系列算法，也是目前强化学习主流算法系列。

论思路主要有以下几条线

- Actor-Critic：包括Actor Critic，A2C,A3C，ACTKR,ACER，SoftAC，SoftAC+adative temperature
- DPG：deterministic policy gradient，包括DPG，DDPG，D4PG，TD3，MADDPG
- TRPO：trust region policy optimization, 包括 TRPO和ACKTR，PPO

## Actor-critic

AC 框架下的算法主要遵循以下思路，Actor用策略梯度更新策略，Critic用值函数之差更新值函数模型。迭代部分步骤是这样的：

```text
 用 A 策略采样current reward和next state
 用 A 策略采样next action
 计算 Q 值得到策略梯度更新 A 策略得到 B 策略（Actor）
 计算值函数差值x策略梯度更新值函数参数（Critic） 
```

**A2C / A3C**

主要是同步和异步的区别。1个actor采样容易出现偏差，A3C套路主要是1个cirtic+n个actor（同一个policy），减少随机性和采用并行化采样，方法和AC一样。异步缺点是由于线程进度不同，这个已经更新了，另一个还没有，可能会同一批actor用的不同policy。A2C 就是加了个 coordinator 来更新policy，能保证大规模数据中更高效。

**Soft AC**

AC中加了个最大熵框架来保证足够随机足够探索。实际上就是把目标函数从最大化reward，加了个，是自适应参数控制比例。也可以用 temperature 的概念，类似于模拟退火。

### **ACER**

actor critic with experience replay ，就是一个 off policy 版本的AC，采样很多，再执行importance sampling (???)用于更新Q，加上为了保持更新后的策略不要离得太远，用运行过的策略参数平均值随便控制一下（没有用 TRPO 或者 PPO 的方法）。

### DPG

DPG　只采用一个动作，可以视为的随机性策略+ AC 框架即可，然后多次采样求期望。如果要增加探索性也可以加入 noise 。

DPG 里面没有找到网络结构。

### DDPG / D4PG / MMDDPG/TD3

DDPG(Deep Deterministic Policy Gradient)=DPG+DQN，DQN 的缺点是在离散空间，DDPG 拓展到了连续空间。

D4PG 就是一个分布式的DDPG，多个 Critic 分布式运行 N step 采样，用 minibatch 期望求Q值，多个 actor 采样到同一个 buffer 里，但是采样是有权重的。

MMDDPG 是一个多 agent 版本的D4PG。只有一个 critic 更新，训练多个 agent 也就是多个 policy，每个 agent 会学习和进化对其他 agent 的值估计，所有 agent 之间进行集成学习得出结果。

TD3 {Scott Fujimoto 2018} 是一种比较特殊的 Q-learning 方法，差不多是 DDPG 的高级trick+ Double Q-learning。用的几个trick有：clipped Double Q-learning，就是用两个critic一起互相借鉴更新Q，而且用的最小值Q保守估计，防止过拟合；delayed update，actor对策略变动频率低于critic；smoothing：考虑到确定性策略典型缺点是在两个值函数有个窄峰的时候容易过拟合，加高斯噪声来随机平滑。TD3 应该是比较好的。

### TRPO / PPO

offpolicy 中一个问题是A（sample policy）、B ( behaviour policy ) 策略不能差别太大，TRPO {John Schulman 2015} 用 KL 散度衡量的话就是两次更新的 KL 距离不能太远，即trust region更新，来保证迭代中性能的单调提升。

PPO 解决距离不能太远的办法没有用 TRPO 这么麻烦的计算，控制策略的概率分布比率在1左右来控制策略相似性。PPO 表现的性能可能比TRPO更好，但是没有特别复杂。

### ACKER

ACTKR用的AC + TRPO的方法。 kronecker factored 大概意思就是为了不让 policy 更新后太远，但也要保持一定距离，用 KL divergence 衡量 policy 使之保持恒定距离。其他的和 AC 一样。



## Ref

[1] Policy Gradient 博客 [https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html](https://link.zhihu.com/?target=https%3A//lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
[2] Richard S Sutton, Andrew G Barto, and others. 1998. *Introduction to reinforcement learning*. MIT press Cambridge.[3] A3C：Mnih, Badia et al. 2016 - Asynchronous Methods for Deep Reinforcement.pdf.
[4] CES：Patryk Chrabaszcz, Ilya Loshchilov et al. 2018 - Back to Basics
[5] DDPG：Timothy P et. al. 2016 Continuous Control With Deep Reinforcement Learning
[6] DPG：David Silver, Guy Lever et al.2014 - Deterministic Policy Gradient Algorithms
[7] DQN：Mnih, Kavukcuoglu et al. 2015 - Human-level control through deep reinforcement
[8] NES：Daan Wierstra, Tom Schaul et al. 2014 - Natural Evolution Strategies
[9] OpenAI ES：Tim Salimans, Jonathan Ho et al. 2017 - Evolution Strategies as a Scalable alternative
[10] PPO：John Schulman, Filip Wolski et al. 2017 - Proximal Policy Optimization Algorithms
[11] TRPO：John Schulman, Sergey Levine et al. 2015 - Trust Region Policy Optimization