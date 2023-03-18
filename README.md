

# Paddle Hackathon 第4期 科学计算

—— 科学计算方向 51

原文：[Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulations of Airfoil Flows](http://arxiv.org/abs/1810.08217)

参考：[Deep-Flow-Prediction Pytorch](https://github.com/thunil/ Deep-Flow-Prediction)

- 本文基于UNet进行了机翼物理场预测，将二维物理场求解过程看作计算机视觉的图像分割任务：首先将几何形状、攻角翼型形状、攻角和Reynolds number转化为输入图像，将物理场（压力、速度场）看作输出结果，采用L1loss进行回归任务训练，文章也详细探讨了不同的网络复杂度、训练策略以及数据增强方法对预测精度的影响，这项研究可以泛化至未知翼型形状中，获得了良好的效果。

- 本文发表于2018年，在当时是比较先进的工作但现在看来，由于低分辨率的像素化处理方式，无法精细化刻画翼型形状，导致了显著的人工粗糙度，对边界层、大梯度的涡量位置无法准确预测，更无法预测准确的气动性能参数，显然不具备实际应用意义。

- 本方法UNet通过学习初始速度分布以及翼型形状，预测在不可压缩Navier-stokes方程（Reynolds-Averaged：Spalart-Allmaras单方程湍流模型）约束下的物理场（压力、速度场），本质属于算子学习的一种，UNet是CNN-based的算子模型，**对标该领域近来较流行的DeepONet、FNO、Transformer等架构**。

- 本次完成UNet、FNO、Transformer的Paddle实现，并将UNet与FNO和Transformer进行了简单对比（考虑DeepONet不对空间关系进行归纳偏置，因此未进行对比），结果也表明FNO和Transformer可获得更高精度，**相对于原文中的相对误差从3%进一步降至2%。**

## 1. 代码说明

- **代码结构**

  ```
  📂 Deep-Flow-Prediction-Paddle
  |_📁 data
    |_📁 train                    # 训练集数据              
      |_📁 reg                    # 常规数据集    
      |_📁 shear                  # 剪切数据集，采用与图像x方向±15°内的扰动产生增强数据集 
      |_📁 sup                    # 高分辨率数据集
    |_📁 test                     # 测试集数据，90个样本     
      |_📄 ***.npz
    |_📁 OpenFOAM                 # OpenFOAM数据集生成
    	|_📁 0                    	# OpenFOAM初始条件和边界条件设定
    	  |_📄 nut                  # 湍流粘度初始/边界条件
        |_📄 nuTilda              # Spalart-Allmaras湍流模型用变量
        |_📄 U                    # 速度初始/边界条件
        |_📄 P                    # 压力初始/边界条件
      |_📁 constant               # OpenFOAM初始条件和边界条件设定
        |_📄 transportProperties	# 流体物性
        |_📄 turbulenceProperties # 湍流模型
      |_📁 system                 # OpenFOAM求解器设置
        |_📄 controlDict          # 求解过程控制
        |_📄 fvSchemes            # fvm离散格式
        |_📄 fvSolution           # 方程组求解方法
        |_📄 internalCloud        # OpenFOAM求解器设置
      |_📄 dataGEN.py             # 数据集生成脚本，需在linux本地机安装openFOAMv9.0后运行
      |_📄 download_airfoils.sh   # uiuc翼型数据集下载
      |_📄 shearAirfoils.py       # 剪切数据集生成脚本
      |_📄 utils.py
    |_📁 pics                     # 原始论文结果相关图片以及作者复现所整理的对应结果
    |_📁 work                     # 训练过程、验证结果、测试结果，统计结果文件保存
      |_📁 transformer            # Results for Transformer
      |_📁 FNO				    # Results for FNO
      |_📁 UNet                   # Results for UNet utilized in original paper
  |_📄 run_plot.py                # 统计结果绘制
  |_📄 run_statistics.py          # 统计验证集损失、测试集误差均值、方差
  |_📄 run_train.py               # 训练过程
  |_📄 run_valid.py               # 测试过程
  |_📄 Trans_model.py             # 二维的Transformer 结构 paddle代码，支持多种attention机制以及两种Regressor
  |_📄 transformer_config.yml     # config for transformer galerkin, fourier, linear, softmax and so on
  |_📄 Unet_model.py              # 原文中实现的CNN-based model paddle代码
  |_📄 FNO_model.py               # 二维的Fourier Neural Operator paddle代码
  ```

- **训练以及测试需要进行如下的参数设置：**

```python
######## Settings ########

# 总体迭代步数，原文采用80k，这里采用200k，注意训练的epoch采用iterations进行运算，以避免不同训练集大小优化步数不同
iterations = 200000
# batch size
batch_size = 10
# learning rate, generator
lrG = 0.0005
# decay learning rate?
decayLr = True
# channel exponent to control network size，注意仅用于UNet，对FNO和Transformer不起作用
expo = 6
# data set config  第一个位置为总体样本数量，后三个数表示从reg/sup/shear三个文件夹选择的样本比例，和为1
# prop = None  # by default, use all from "../data/train"
prop = [10000, 1.0, 0, 0.0]  
# save txt files with per epoch loss?
saveL1 = True
# model type UNet/FNO/Transformer
net = 'Transformer'
# statistics number  每个训练过程执行次数，以便在run_valid和run_staticstics进行统计多次的均值和方差，绘制Fig10 11
sta_number = 10
# data path
data_path = os.path.join('data')

```

- **执行过程**
  - **运行run_train，时间较久，batch_size=10，nvidia 3090测试200k对UNet expo=7 （参数量30.9m）约需要1小时**
  - **运行run_valid，获得test集合上的90个case的结果云图以及各个训练过程的test relative error**
  - **运行run_statistics，获得valid loss 和 test relative error统计结果均值及方差**
  - **运行run_plot，对所有统计结果均值及方差进行绘图，文件中给出了实验过程的统计结果文件**
  - **考虑训练时间过久,为快速起见，可先设置sta_number=1，跑出一组查看效果**

- **环境依赖**

  > numpy == 1.22.3 \
  > paddlepaddle-gpu develop \
  > matplotlib==3.5.1 \



## 2. 原始数据集

数据范围： 从[UIUC数据集SoIaUCAD96](https://m-selig.ae.illinois.edu/ads/coord_database.html)获取的1505个翼型， Re=[0.5, 5]×10<sup>6</sup> , 攻角 [-22.5, 22.5]deg，原文给出的训练集样本总计约53,830个，测试集90个。

数据获取：OpenFOAM, 湍流模型SA，数值计算网格划分尺寸为1/200（机翼弦长固定为1， 计算域尺寸为8×8），深度学习关注域为机翼周围2×2，并重新将物理场插值为128×128的矩阵

数据描述：

|      | 数据格式      | 数据组成                                                     |
| ---- | ------------- | ------------------------------------------------------------ |
| 输入 | （128×128）×3 | 由两部分组成，部分为网格结构下的机翼几何mask $ \phi =0 $ 表示在机翼外部， $\phi =1$ 表示在机翼内部；第二部分为  由Re和攻角决定的初始速度场 $v_i = {v_{i,x}, v_{i,y}}$ |
| 输出 | （128×128）×3 | 网格结构下的机翼周围压强、x-方向和y-方向速度场  ${p_o, v_{o,x}, v_{o,y}}$ |

数据预处理： 首先，采用来流条件进行速度和压强归一化，对于速度而言 
$$
\tilde{v}_ o = v_o/|v_i|
$$
对于压强而言  
$$
\tilde{p}_ o = v_o/|v_i|^2
$$
其次，对压强进行二次处理， 
$$
\hat{p}_ {o} = \tilde{p}_ o - p_{mean}
\\ p_{mean}=\sum_{i}p_i/n
$$
最后，采用最大最小值将物理场标准化至[-1, 1]以最小化有限数值精度的误差。

## 3. 网络架构以及训练方法

### a. 网络架构

文章采用视觉中常用的UNet结构，在论文复现过程中，本文同时对比了Fouriour Neural operation和Transformer，详见第5部分。
UNet 构架如下图所示，它可以视为一个编码器-解码器框架，在编码器部分，图片尺寸按照2的比例采用卷积（stride=2）逐渐下采样，提取特征；在解码器部分，进行镜像操作，逐渐增大图片尺寸，而减少通道数。在对应图片尺寸之间进行跳跃连接以帮助解码器了解编码器过程中的低阶信息。对于7.7m的UNet,编码器中 采用7个卷积block将（128×128）×3的输入转换为512个特征，卷积核尺寸为（4×4），激活函数为ReLU（Slope=0.2）;解码器中采用对称的7个结构（采用标准的ReLU）重构为输出（128×128）×3。
![UNet-arch](https://ai-studio-static-online.cdn.bcebos.com/bd153e3f09ae41e090144920143259757923b909af584cd3b1f3cda2c56829f3)

- **注意：UNet网络结构参数量通过expo控制，FNO、Transformer需要自行设置，使用者根据相关文献自行选择。**

```python
epochs = int(iterations / len(trainLoader) + 0.5)
if 'UNet' in net:
    net_model = UNet2d(channelExponent=expo, dropout=dropout)
elif 'FNO' in net:
    net_model = FNO2d(in_dim=3, out_dim=3, modes=(32, 32), width=32, depth=4, steps=1, padding=4,
                      activation='gelu')
elif 'Transformer' in net:
    import yaml

    with open(os.path.join('transformer_config.yml')) as f:
        config = yaml.full_load(f)
    config = config['Transformer']
    net_model = FourierTransformer2D(**config)
```

### b. 训练方法

三种网络框架采用完全相同的训练方法，即 Adam 优化器， 统一训练200K个迭代步，损失函数采用 $L_1$损失：
损失函数的具体公式为

| 损失函数   | 公式                                                         |
| ---------- | ------------------------------------------------------------ |
| x-方向速度 | $$ L_{vx}=\sum_{i}\|\tilde{v}_ {o,x} - \tilde{v}_ {pred, o, x}\| $$ |
| x-方向速度 | $L_{vy}=\sum_{i}\|\tilde{v}_ {o,y} - \tilde{v}_ {pred, o,y} \|$ |
| 压强       | $L_{p}=\sum_{i}\|\hat{p}_ {o} - \hat{p}_ {pred, o} \|$       |
| 总损失函数 | $L_{total}= (L_p + L_{vx} + L_{vy})/3$                       |

## 4. 复现结果

**复现Fig8** 从测试集中选择若干样例展示

| 0039                                                         | **086**                                                      | 010                                                          | 017                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![expo=7 jet 086](https://ai-studio-static-online.cdn.bcebos.com/2af2491fb15b40e092b797325cd891ae4b7cff74557a4c35b5181935d010a87f) | ![expo=7 jet 039](https://ai-studio-static-online.cdn.bcebos.com/f34981bca80a418c87945612ad7cee14943cf924b14841a89d84a589e2acbb98) | ![expo=7 jet 010](https://ai-studio-static-online.cdn.bcebos.com/e711b74f088f40b8b677781f60e869ae50623c4e642545a4b850cd6772cfff7d) | ![expo=7 jet 017](https://ai-studio-static-online.cdn.bcebos.com/ea8622a27659453bae52a5a2dce50361da4dc32685d240ebb7b1970567685d84) |

**论文Fig8**

![Fig8-](https://ai-studio-static-online.cdn.bcebos.com/b36597c14c6641b795530cf9ede6f34243b25c91a44041a48edc03a96bb3a726)

**复现Fig9** 从测试集中选择0079编号的case在不同网络参数下的结果展示

| 参数量 | 1.9m（expo=5）                                               | 7.7m（expo=6）                                               | 30.9m（expo=7）                                              |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Fig9   | ![expo=7 0079](https://ai-studio-static-online.cdn.bcebos.com/d226539ee467484cbcb3d0af0cae9a9765d2214e60b04308ad0b06f6a78d715d) | ![expo=6 0079](https://ai-studio-static-online.cdn.bcebos.com/3620853d5146413dab268e600a2707e8b42f77e081344e2ea669e04af3518b9a) | ![expo=5 0079](https://ai-studio-static-online.cdn.bcebos.com/ae3f7dac197e4200adc97b2b752637b30f615459ca164ed1be506c58e0cc0c09) |

**论文Fig9**

![Fig9-](https://ai-studio-static-online.cdn.bcebos.com/3aed5383003f4698a5378d067d875de697c3ce64f5a945008375319d96f8bc8f)

| 统计结果 |  复现（计算10次统计结果）  | 论文 |
| :--: | :------------: | :------: |
| Fig10 |![Fig10](https://ai-studio-static-online.cdn.bcebos.com/33344a7316434374b4f977d41dcb33da9a0cfab4644e40018bb69acfc3fce38e)| <img src="https://ai-studio-static-online.cdn.bcebos.com/3a1d7dd7c75e43b09ea7091bc88e059853bafcddd2b44f178456802085f2fbbf" alt="Fig10-" style="zoom:250%;" /> |
| Fig11 |![Fig11](https://ai-studio-static-online.cdn.bcebos.com/a066fe37138e4e66ad9d3e00fa14d432262bea787b9d46b3b17d26777363f67e)|<img src="https://ai-studio-static-online.cdn.bcebos.com/bae5a16083564e3589372d63937aae6c0a73a0e521cd4a9b8bd3d49484811852" alt="Fig11-" style="zoom:250%;" />|




## 5.拓展实现

**FNO (Fourier Neural Operator)**  

reference：[Fourier Neural Operator for Parametric Partial Differential Equations](http://arxiv.org/abs/2010.08895)

github：https://github.com/zongyi-li/fourier_neural_operator

![](https://ai-studio-static-online.cdn.bcebos.com/cfe9cb2617254433b78dbb6c144809e6f4b42bc85f8748cb945602e6ee424847)

Fourier Spectrual neural operator 核心算法实现过程：

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

**Garlerkin Transformer**

reference：[Choose a Transformer: Fourier or Galerkin](http://arxiv.org/abs/2010.08895)

github：https://github.com/scaomath/galerkin-transformer

本文中提出了与原始attention机制的改进方法，类比有限元方法，近似线性化降低了计算复杂度，提升了计算效率。网络的详细设置

![Trans-arch](https://ai-studio-static-online.cdn.bcebos.com/5afd294cfd6447189d5c655870f2d0706a05815e03944380835c44b70008d245)

| 形式     | attention计算方式                                            |                                                              |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fourier  | ![Trans-fourier](https://ai-studio-static-online.cdn.bcebos.com/a62efc6b06d3441492c507b1c381521fe474fcf9eed94ee6b53a7c36cd24184b) | <img src="https://ai-studio-static-online.cdn.bcebos.com/e1efb1c56a0c44ef8b32d8c4d2e8e3c529e70800c13e4547aceafefa440bfe8a" alt="Trans-fourier_" style="zoom:50%;" /> |
| galerkin | ![Trans-galerkin](https://ai-studio-static-online.cdn.bcebos.com/d23dcfa98ce34bf9a22d9a471a5bb8cf370e064dc53248fcbe0dae87b1e6c0cf) | <img src="https://ai-studio-static-online.cdn.bcebos.com/e1efb1c56a0c44ef8b32d8c4d2e8e3c529e70800c13e4547aceafefa440bfe8a" alt="Transformer_galerkin_" style="zoom:50%;" /> |

改进的attention机制核心算法实现过程：（包括fourier,galerkin,linear,softmax）

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



## 6. 三种网络架构对比

在25600的训练样本下，三种网络分别训练iterations=100k，结果如图

| Model | UNet                                                         | FNO                                                          | Transformer-galerkin                                         |
| ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0079  | ![img](https://ai-studio-static-online.cdn.bcebos.com/d226539ee467484cbcb3d0af0cae9a9765d2214e60b04308ad0b06f6a78d715d) | ![FNO](https://ai-studio-static-online.cdn.bcebos.com/e551ee5aac514f2bbad002d6ed5b6e6f00312b3118c7459cac0c7e7d2a10d058) | ![Trans](https://ai-studio-static-online.cdn.bcebos.com/630ffc91bedf47bb888764209b6beb6c220badb40e834ac081418524a12c9990) |
| 0004  | ![UNet004](https://ai-studio-static-online.cdn.bcebos.com/24b1b935d83347c3b9ee8f4dc9a19e761c5ee99d366b4470a9936a5c20ed591b) | ![FNO004](https://ai-studio-static-online.cdn.bcebos.com/fbb9366b69004e22b5d268a7b7f270dbc8caea1b3c514c45b3d962bbe7b7df1e) | ![Trans004](https://ai-studio-static-online.cdn.bcebos.com/97aaefedb013417d979a1d4fd5b1aec0165d91abd7e94e96adc4e2dc00715282) |
| 指标  | Loss percentage (p, v, combined): 24.304499 %    4.075242 %    5.419715 % L1 error: 0.011459<br/>Denormalized error: 0.032662 | Loss percentage (p, v, combined): 9.744780 %    1.717354 %    2.072256 % L1 error: 0.004192<br/>Denormalized error: 0.009566 | Loss percentage (p, v, combined): 9.847503 %    1.739543 %    2.105041 % <br/>L1 error: 0.004261<br/>Denormalized error: 0.010268 |

不同训练集大小下，运行10次统计结果：

![comp](https://ai-studio-static-online.cdn.bcebos.com/7ac5d8377d374209a4b89ed8b5924afdb671902e78cd48369bac660e2a0e26e9)

本次测试测试结果进行了一个简单的对比，FNO和Transformer优于UNet，也优于原始论文中的最佳值，但这个结论值得进一步讨论。事实上，本次完成的Transformer中提供了多种attention机制，以及不同的Regressor，对于FNO和Transformer以及更多算子学习方法的速度、精度和稳定性的对比可以在这个标准数据集上展开，完成类似[PDEBench](http://arxiv.org/abs/2210.07182)中的工作。

**本次代码仅仅抛砖引玉，期待有兴趣的使用者在此基础上进行进一步探索。**

## 7. 模型信息

| 信息          | 说明                                                      |      |
| ------------- | --------------------------------------------------------- | ---- |
| 发布者        | tianshao1992                                              |      |
| 时间          | 2023.3                                                    |      |
| 框架版本      | Paddle Develope                                           |      |
| 应用场景      | 科学计算                                                  |      |
| 支持硬件      | CPU、GPU                                                  |      |
| AI studio地址 | https://aistudio.baidu.com/aistudio/projectdetail/5584432 |      |
