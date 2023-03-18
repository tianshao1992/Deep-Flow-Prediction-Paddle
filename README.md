

# Paddle Hackathon 第4期 科学计算

—— 科学计算方向 51

原文：[Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations of Airfoil Flows](http://arxiv.org/abs/1810.08217)

参考：[Deep-Flow-Prediction Pytorch](https://github.com/thunil/ Deep-Flow-Prediction)

- 算子学习

- 在GPINNs的实现过程中，作者将PDE残差的梯度信息作为一个加权损失函数补充训练。相对于传统的PINNs，作者处理的技巧主要包括了两点（GPINNs——Gradient-enhanced physics-informed neural networks）：

  - 在损失函数设计，GPINNs引入了PDE残差对各个自变量的梯度残差损失，并采用加权损失函数。
  - 所有问题中均引入了[hard constraints](https://epubs.siam.org/doi/10.1137/21M1397908)约束神经网络输出，使得边界条件和初始条件天然满足，降低了神经网络优化的难度。
  - 在训练过程中，先以大量的残差样本点作为初始训练集，在训练过程中，通过不断寻找PDE残差较大的区域进行重采样进行补充训练集——即类似主动学习的策略对PINNs进行训练。


## 2.代码说明

1. [gPINNs4paddle AI studio](https://aistudio.baidu.com/aistudio/projectdetail/4493662)相关运行结果

  - 脚本文件包括

    - run_train.py——训练文件。
    - run_valid.py——测试文件。
    - 训练以及测试需要进行如下的参数设置：

    ```python
    ######## Settings ########
    
    # number of training iterations
    iterations = 200000
    # batch size
    batch_size = 10
    # learning rate, generator
    lrG = 0.0005
    # decay learning rate?
    decayLr = True
    # channel exponent to control network size
    expo = 6
    # data set config
    # prop = None  # by default, use all from "../data/train"
    prop = [10000, 1.0, 0, 0.0]  # mix data from multiple directories
    # save txt files with per epoch loss?
    saveL1 = True
    # model type UNet/FNO/Transformer
    net = 'Transformer'
    # statistics number
    sta_number = 10
    # data path
    data_path = os.path.join('data')
    
    ```

    其中，iterations可供选择采用gPINN模式或pinn模式；

    batch_size为采用adam优化器进行训练的总步数；

    lrG为打印残差以及输出中间可视化结果的频率；

    decayLr为是否使用gpu并行；

    expo为UNet的网络层数，对于FNO以及Transformer无效。

    prop为训练PDE残差的采样点个数；

    saveL1分别指代采用验证数据时在时间t（如果存在）以及x方向的采样点个数（均匀网格）；

    net为gradient ehanced的残差权重

    sta_number为将当前训练过程跑若干次，以统计结果均值、方差

  - **pics文件夹**中为原始论文结果相关图片以及作者复现所整理的对应结果，**work文件夹**中为训练过程及中间结果

  - UNet_model.py 为原文中实现的CNN-based model paddle代码

  - FNO_model.py 中为二维的Fourier Neural Operator paddle代码

  - Trans_model.py 中为二维的Transformer 结构 paddle代码，支持多种attention机制以及两种Regressor

## 3.环境依赖

  > numpy == 1.22.3 \
  > paddlepaddle-gpu develop \
  > matplotlib==3.5.1 \

## 4.复现结果

### 4.1 正向问题 

1).原始方程为1-D Poisson方程：

$$
-\Delta u=\sum_{i=1}^4 i \sin (i x)+8 \sin (8 x), \quad x \in[0, \pi]
$$

方程解析解为

$$
u(x)=x+\sum_{i=1}^4 \frac{\sin (i x)}{i}+\frac{\sin (8 x)}{8}
$$

损失函数为

$$
\mathcal{L}=\mathcal{L}_f+w \mathcal{L}_g
$$

1D Possion （对应文章 3.2.1 Figure 2）详细代码见run_3.2.1.py 以及train_3.2.1.sh 设置不同的traning points——Nx_EQs以及权重 $w$ 。（误差带为运行10次求取均值以及方差绘制，以下类似。）
下表详细展示了采用GPINN对1D Possion问题的预测效果，以及不同权重、不同训练点数量对于GPINN的影响。其中，左侧为本次复现结果，而右侧为论文结果。需要指出，复现的结果中PINNs以及gPINNs均较原始论文更好，其中GPINNs 权重为1.0时效果尤其明显。

**复现Fig8**

| 0039                                                         | **086**                                                      | 010                                                          | 017                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![expo=7 jet 086](https://ai-studio-static-online.cdn.bcebos.com/2af2491fb15b40e092b797325cd891ae4b7cff74557a4c35b5181935d010a87f) | ![expo=7 jet 039](https://ai-studio-static-online.cdn.bcebos.com/f34981bca80a418c87945612ad7cee14943cf924b14841a89d84a589e2acbb98) | ![expo=7 jet 010](https://ai-studio-static-online.cdn.bcebos.com/e711b74f088f40b8b677781f60e869ae50623c4e642545a4b850cd6772cfff7d) | ![expo=7 jet 017](https://ai-studio-static-online.cdn.bcebos.com/ea8622a27659453bae52a5a2dce50361da4dc32685d240ebb7b1970567685d84) |

**论文Fig8**

![Fig8-](https://ai-studio-static-online.cdn.bcebos.com/b36597c14c6641b795530cf9ede6f34243b25c91a44041a48edc03a96bb3a726)

**复现Fig9**

| 参数量 | 1.9m                                                         | 7.7m                                                         | 30.9m                                                        |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Fig9   | ![expo=7 0079](https://ai-studio-static-online.cdn.bcebos.com/d226539ee467484cbcb3d0af0cae9a9765d2214e60b04308ad0b06f6a78d715d) | ![expo=6 0079](https://ai-studio-static-online.cdn.bcebos.com/3620853d5146413dab268e600a2707e8b42f77e081344e2ea669e04af3518b9a) | ![expo=5 0079](https://ai-studio-static-online.cdn.bcebos.com/ae3f7dac197e4200adc97b2b752637b30f615459ca164ed1be506c58e0cc0c09) |

**论文Fig9**

![Fig9-](https://ai-studio-static-online.cdn.bcebos.com/3aed5383003f4698a5378d067d875de697c3ce64f5a945008375319d96f8bc8f)

|      |  复现  | 论文 |
| :--: | :------------: | :------: |
| Fig10 |![Fig10](https://ai-studio-static-online.cdn.bcebos.com/33344a7316434374b4f977d41dcb33da9a0cfab4644e40018bb69acfc3fce38e)| <img src="https://ai-studio-static-online.cdn.bcebos.com/3a1d7dd7c75e43b09ea7091bc88e059853bafcddd2b44f178456802085f2fbbf" alt="Fig10-" style="zoom:250%;" /> |
| Fig11 |![Fig11](https://ai-studio-static-online.cdn.bcebos.com/a066fe37138e4e66ad9d3e00fa14d432262bea787b9d46b3b17d26777363f67e)|<img src="https://ai-studio-static-online.cdn.bcebos.com/bae5a16083564e3589372d63937aae6c0a73a0e521cd4a9b8bd3d49484811852" alt="Fig11-" style="zoom:250%;" />|




## 6.拓展实现

FNO (Fourier Neural Operator)  

reference：[Fourier Neural Operator for Parametric Partial Differential Equations](http://arxiv.org/abs/2010.08895)

github：https://github.com/zongyi-li/fourier_neural_operator

![](https://ai-studio-static-online.cdn.bcebos.com/cfe9cb2617254433b78dbb6c144809e6f4b42bc85f8748cb945602e6ee424847)

```python

class SpectralConv2d(nn.Layer):
    '''
    2维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: tuple,  # number of fourier modes
                 dropout=0.0,
                 norm='ortho',
                 activation='gelu',
                 return_freq=False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        if isinstance(modes, int):
            self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes
        else:
            self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes[1]

        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_dict[activation]
        self.return_freq = return_freq
        self.linear = nn.Conv2D(self.in_dim, self.out_dim, 1)  # for residual

        self.weights1 = paddle.create_parameter([in_dim, out_dim, self.modes1, self.modes2, 2], dtype=paddle.float32)
        w1 = params_initial('xavier_normal', shape=self.weights1.shape,
                            gain=1 / (in_dim * out_dim) * np.sqrt(in_dim + out_dim))
        self.weights1.set_value(w1)
        self.weights2 = paddle.create_parameter([in_dim, out_dim, self.modes1, self.modes2, 2], dtype=paddle.float32)
        w2 = params_initial('xavier_normal', shape=self.weights2.shape,
                            gain=1 / (in_dim * out_dim) * np.sqrt(in_dim + out_dim))
        self.weights2.set_value(w2)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        op = partial(paddle.einsum, "bixy,ioxy->boxy")
        return paddle.stack([
            op(input[..., 0], weights[..., 0]) - op(input[..., 1], weights[..., 1]),
            op(input[..., 1], weights[..., 0]) + op(input[..., 0], weights[..., 1])
        ], axis=-1)

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.linear(x)
        x = self.dropout(x)
        x_ft = paddle.fft.rfft2(x, norm=self.norm)
        x_ft = paddle.stack([x_ft.real(), x_ft.imag()], axis=-1)

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros((batch_size, self.out_dim, x.shape[-2], x.shape[-1] // 2 + 1, 2), dtype=paddle.float32)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = paddle.complex(out_ft[..., 0], out_ft[..., 1])

        # Return to physical space
        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]), norm=self.norm)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x
```

Garlerkin Transformer

reference：[Choose a Transformer: Fourier or Galerkin](http://arxiv.org/abs/2010.08895)

github：https://github.com/scaomath/galerkin-transformer



![Trans-arch](https://ai-studio-static-online.cdn.bcebos.com/5afd294cfd6447189d5c655870f2d0706a05815e03944380835c44b70008d245)

| 形式     | attention计算方式                                            |      |
| -------- | ------------------------------------------------------------ | ---- |
| fourier  | ![Trans-fourier](https://ai-studio-static-online.cdn.bcebos.com/a62efc6b06d3441492c507b1c381521fe474fcf9eed94ee6b53a7c36cd24184b) |      |
| galerkin | ![Trans-galerkin](https://ai-studio-static-online.cdn.bcebos.com/d23dcfa98ce34bf9a22d9a471a5bb8cf370e064dc53248fcbe0dae87b1e6c0cf) |      |



```python

class SimpleAttention(nn.Layer):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types:
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = paddle.einsum('bhnd,bhne->bhde', k, v)
    attn = paddle.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm_add=False,
                 norm_type='layer',
                 eps=1e-5):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0  # n_head 可被d_model整除
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.LayerList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.norm_add = norm_add
        self.norm_type = norm_type
        if norm_add:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        """
        forward compute
        :param query: (batch, seq_len, d_model)
        :param key: (batch, seq_len, d_model)
        :param value: (batch, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.shape[0]
        if weight is not None:
            query, key = weight * query, weight * key

        query, key, value = \
            [layer(x).reshape((bsz, -1, self.n_head, self.d_k)).transpose((0, 2, 1, 3))
             for layer, x in zip(self.linears, (query, key, value))]

        if self.norm_add:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose((0, 1, 3, 2)), value.transpose((0, 1, 3, 2))

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                value = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose((0, 1, 3, 2)), value.transpose((0, 1, 3, 2))
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose((0, 1, 3, 2)), query.transpose((0, 1, 3, 2))

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                query = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose((0, 1, 3, 2)), query.transpose((0, 1, 3, 2))

        if pos is not None and self.pos_dim > 0:
            assert pos.shape[-1] == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.tile([1, self.n_head, 1, 1])
            query, key, value = [paddle.concat([pos, x], axis=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        else:
            x, self.attn_weight = vanilla_attention(query, key, value,
                                                    mask=mask,
                                                    attention_type=self.attention_type,
                                                    dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
                                                             (self.d_k + self.pos_dim)
        att_output = x.transpose((0, 2, 1, 3)).reshape((bsz, -1, out_dim))

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

```



## 7.三种网络架构对比

| Model | UNet                                                         | FNO                                                          | Transformer-galerkin                                         |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0079  | ![img](https://ai-studio-static-online.cdn.bcebos.com/d226539ee467484cbcb3d0af0cae9a9765d2214e60b04308ad0b06f6a78d715d) | ![FNO](https://ai-studio-static-online.cdn.bcebos.com/e551ee5aac514f2bbad002d6ed5b6e6f00312b3118c7459cac0c7e7d2a10d058) | ![Trans](https://ai-studio-static-online.cdn.bcebos.com/630ffc91bedf47bb888764209b6beb6c220badb40e834ac081418524a12c9990) |
| 0004  | ![UNet004](https://ai-studio-static-online.cdn.bcebos.com/24b1b935d83347c3b9ee8f4dc9a19e761c5ee99d366b4470a9936a5c20ed591b) | ![FNO004](https://ai-studio-static-online.cdn.bcebos.com/fbb9366b69004e22b5d268a7b7f270dbc8caea1b3c514c45b3d962bbe7b7df1e) | ![Trans004](https://ai-studio-static-online.cdn.bcebos.com/97aaefedb013417d979a1d4fd5b1aec0165d91abd7e94e96adc4e2dc00715282) |



![comp](https://ai-studio-static-online.cdn.bcebos.com/7ac5d8377d374209a4b89ed8b5924afdb671902e78cd48369bac660e2a0e26e9)



## 7.模型信息

| 信息          | 说明                                                      |      |
| ------------- | --------------------------------------------------------- | ---- |
| 发布者        | tianshao1992                                              |      |
| 时间          | 2023.3                                                    |      |
| 框架版本      | Paddle Develope                                           |      |
| 应用场景      | 科学计算                                                  |      |
| 支持硬件      | CPU、GPU                                                  |      |
| AI studio地址 | https://aistudio.baidu.com/aistudio/projectdetail/5584432 |      |
