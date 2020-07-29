# 20200729 科研新知｜图神经网络的研究图景

2020 07 29 读完第一遍GNN reveiw 和Zhiyuan Liu Introduction of GNN的大致感知

GNN 领域是一个基于现有的神经网络和机器学习理论在非欧空间上的 extension ，所以大量图神经网络的基础部分与机器学习相似，一类工作是想办法通过类比（如卷积概念、局部性）把原有 CNN 的大量研究迁移到图神经网络上，另一类是直接从图的特性出发，研究图神经网络特有的难点如计算量梯度运算。因此，对比 CNN ，可以从1）学习任务（监督学习、非监督学习等）、2）训练方法、3）网络结构等几个维度对近年研究分类。由于图这种数据结构与欧氏空间的数据相比，具备更多属性和变种，因此根据4）图类型也是一种维度。

所以，根据最原始的 GNN 模型延伸可以分为

1. Learning task

    这里所有任务又分为 节点、边、全局图三类

    1）Supervised：

    - Node classification and node regression

        如推荐系统对用户类型分类、用户属性预测，社交网络中对关键节点的辨识和分类

    - Edge classification and link prediction

        如社交网络中关系预测、关系推断，知识图谱关系推理预测，

    - Graph classification

        如物理系统的系统状态辨识

    2) Semi-supervised:

    - Node labelling in partial labelled graph

        与监督学习类似

    - Edge labelling 

    3) Unsupervised:

    - Graph embedding : 如何用低维向量化重构原图
    - Genearative model : 生成对抗式问题，生成图等

2. Training technique

    1）matrix computation：涉及到完全 Laplacian 矩阵运算节点增加时计算量指数增长，通过约减矩阵、近似矩阵、低秩压缩矩阵等系列方法解决这个问题的一系列方法

    2）gradient computation：由于节点嵌入是根据上一层的节点递归运算，感受野会越来越大，单个节点的梯度运算计算量突增，解决梯度运算/或采样感受野/减少层数的一系列研究

    3）dataset：达到深度学习需要的数据量相当的图，输入数据相当庞大并不现实，因此针对数据集的高效利用和数据增强技术也有一系列研究

    没有太多已发表文献但当前研究比较热门的问题有：

    1）over smoothing ：节点倾向于收敛为一个值，使得网络深度不能加深，影响学习效果

    2)  over fitting：对单个图进行学习，不能归纳，在随时可能加入节点、边以及关系变化的动态图上泛化能力还比较差

    3）scalability : 大规模复杂网络的学习，特别是拓展到社交网络这个层级的，和1）相比规模更大，可能需要多种方法联合使用，如图嵌入先缩小到可处理的低维向量再用 1）

3. Network Architecture

    按照图神经网络所使用的网络结构类型分大致是5类。

    1）GCN ：Graph convolution Neural Network

    - Spatial methods : 主要区别是如何定义和选择 neighborhood 来迁移 CNN 的卷积操作
    - Spectral methods：空域特例，直接在图的 Laplacian matrix 上定义傅立叶变换和卷机操作迁移

    2）GRN：Graph Recurrent Neural Network

    3）GAN：Graph Attention Neural Network

    4）GAE：Graph Autoencoders

    4）GRN（？）：Graph Residual Networks

4. Graph type

    1）Directed

    2)  Heterogeneous 

    3)  Edge Informational

    4) Dynamic Graphs：目前研究较少的一个难点，当前网络均为静态输入的情况下，对于随时动态调整的图结构没什么泛化性能。

    5) Multi-dimensional Graphs 

5. Framework

    以上方法均相对独立，对这些方法兼容性最好的 GNN 框架包括以下三种。整体来说都是两阶段式，传播阶段（不同论文叫法包括propagation phase/message passing/computation steps/aggeration）和输出阶段（叫法包括output/readout/activate）。

    1）传播阶段： fixed theorem 理论证明图通过邻域节点传播一定步后会收敛到稳定状态，最后稳定状态的隐状态$H  = F(H,X)$

    2）输出阶段：根据隐状态求输出标签/预测值 $O = G(H,X)$

    特别是第三种，把图模型分为节点集、边集、全局集，分六步更新，不同的结构和传播训练方式都可以兼容，可能可以根据该框架找领域空白填补。

 - message passing neural network
 - non local neural network
 - graph network

