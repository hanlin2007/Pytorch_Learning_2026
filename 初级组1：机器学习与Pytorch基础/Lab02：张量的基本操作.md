
>对于张量部分，我采用的学习方式是快速学习官方教程的笔记，同时让AI给出了一个“实用的速成清单”，以掌握和能够应用为目的，而不是说非要死记硬背多少语法


#### 1. 为什么必须先学这个？

深度学习的核心，就是**把数据（图片、文字）变成数字，然后进行海量的数字运算**。

在Python标准库中，处理数字的通常是列表（List），但它的运算速度太慢。NumPy虽然很快，但无法在GPU（显卡）上运行。而深度学习的计算量巨大，必须用GPU并行处理。

因此，PyTorch的核心数据结构诞生了：**张量（Tensor）**。你可以把它简单理解为**能在GPU上高速运转的NumPy数组**。整个PyTorch框架和深度学习模型，本质上就是在学习如何对Tensor进行各种变换。

#### 2. Tensor到底是什么？

-   **0维张量（标量）：** 就是一个单独的数字，比如 `3.14`。
-   **1维张量（向量）：** 一排数字，比如 `[1.0, 2.0, 3.0]`。可以想象成一条线上的点。
-   **2维张量（矩阵）：** 一个表格，比如 `[[1,2], [3,4]]`。有行和列，可以想象成一张黑白图片的像素网格。
-   **3维张量：** 多个矩阵叠在一起。比如一张彩色图片，可以看作是高度 `H`、宽度 `W` 的矩阵，但有RGB三个颜色通道，所以它的维度就是 `(3, H, W)`。
-   **4维张量：** 在3维的基础上，再加一个“样本数量”的维度 `N`。我们在训练时一次看多张图片，数据的维度就是 `(N, C, H, W)`，这是计算机视觉中非常经典的格式。

#### 3. PyTorch实战：核心代码与技巧

**第一步：创建Tensor**

这是你需要熟练掌握的几种方式：

```python
import torch
import numpy as np

# 1. 从Python列表直接创建 (最常用)
a = torch.tensor([[1, 2], [3, 4]])
print(a)

# 2. 创建特殊张量 (初始化常用)
zeros = torch.zeros(2, 3)      # 2行3列，全0
ones = torch.ones(2, 3)        # 全1
eye = torch.eye(3)             # 3x3单位矩阵
rand = torch.rand(2, 3)        # 均匀分布 [0,1) 的随机数
randn = torch.randn(2, 3)      # 标准正态分布随机数 (均值为0，方差为1)

# 3. 从NumPy转换 (处理数据时常用)
np_array = np.array([[5, 6], [7, 8]])
tensor_from_np = torch.from_numpy(np_array)

# 4. 查看Tensor的属性
print(rand.shape)    # 形状: torch.Size([2, 3])
print(rand.dtype)    # 数据类型: torch.float32
print(rand.device)   # 所在设备: cpu
```

**第二步：Tensor的运算——形状变换（最重要的部分）**

在搭建神经网络时，数据流经不同层，形状会不断改变。掌握形状变换是调试模型的基础。

-   **`view()` 或 `reshape()`：** 重新排列元素，改变Tensor的形状。
    -   可以把它想象成把一堆积木重新堆叠。`-1`是一个“自适应”参数，PyTorch会自动计算这个维度的大小。

```python
x = torch.randn(2, 3, 4)  # 一个2x3x4的张量
print(x.shape)

# 把它展平成2维： (2, 12)。 12 = 3*4
x_2d = x.view(2, -1)      # -1 告诉PyTorch自动计算第二个维度
print(x_2d.shape)         # 输出: torch.Size([2, 12])

# 把它展平成1维： (24)
x_1d = x.view(-1)
print(x_1d.shape)         # 输出: torch.Size([24])
```

-   **`unsqueeze()` 和 `squeeze()`：** 增加或减少维度。
    -   `unsqueeze`：在某一个位置增加一个大小为1的维度。比如把一排数字 `[1,2,3]`（3,）变成一个列向量 `[[1],[2],[3]]`（3, 1）。
    -   `squeeze`：去掉所有大小为1的维度。

```python
x = torch.randn(3)        # 形状: (3,)
print(x.unsqueeze(0).shape) # 在0维增加一个维度: (1, 3)
print(x.unsqueeze(1).shape) # 在1维增加一个维度: (3, 1)
```

**第三步：Tensor的运算——数学计算**

-   **四则运算：** `+`, `-`, `*`, `/` 都是**元素级别**的运算。
-   **矩阵乘法（这是神经网络的核心）：** 使用 `@` 或 `torch.matmul()` 或 `.mm()`。
    -   想象一下，神经网络的“学习”就是找到一组权重矩阵W，当输入数据X经过 `X @ W` 这个矩阵乘法后，能得到我们想要的结果。

```python
# 矩阵乘法示例
A = torch.randn(3, 4)    # 3行4列
B = torch.randn(4, 5)    # 4行5列
C = torch.matmul(A, B)   # 结果C的形状是 (3, 5)
print(C.shape)
```

**第四步：Tensor的移动（CPU <-> GPU）**

深度学习没有GPU基本无法进行。你要习惯把模型和数据都放到GPU上。

```python
# 检查是否有GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available, using GPU.")
else:
    device = torch.device('cpu')
    print("GPU not available, using CPU.")

# 创建一个Tensor
x = torch.randn(2, 3)

# 将Tensor移动到GPU
x_gpu = x.to(device)

# 或者创建时直接指定设备
x_gpu_direct = torch.randn(2, 3, device=device)

# 后续运算x_gpu就会在GPU上进行，速度飞快
```

#### 4. 总结与思考

在Lab-01阶段，你不需要死记硬背所有函数，只需要建立两个核心概念：

1.  **数据即Tensor：** 任何输入、输出、中间结果，在PyTorch里都是Tensor。
2.  **形状即结构：** 时刻关注并理解你手中Tensor的 `shape` 属性。**当程序报错时，90%是因为形状不匹配**。学会用 `view()` 来调整形状，解决大部分问题。


# PyTorch 张量简介#

 [YouTube](https://www.youtube.com/watch?v=r7QDUPb2dCM) 视频链接

张量是 PyTorch 的核心数据抽象。本交互式笔记本深入介绍了 `torch.Tensor` 班级。

首先，我们导入 PyTorch 模块。我们还会添加 Python 的 math 模块，以便于演示一些示例。

```
import torch
import math
```

## 创建张量#

创建张量的最简单方法是使用 `torch.empty()` 调用：

```
x = torch.empty(3, 4)
print(type(x))
print(x)
```


```
<class 'torch.Tensor'>
tensor([[-9.9359e-12,  4.5630e-41, -7.2680e-12,  4.5630e-41],
        [-1.0034e-11,  4.5630e-41, -6.9037e-12,  4.5630e-41],
        [ 4.0228e+09,  4.5629e-41,  6.6955e+09,  4.5629e-41]])
```



让我们总结一下刚才做的事情：

- 我们使用 `torch` 模块附带的众多工厂方法之一创建了一个张量。
- 该张量本身是二维的，有 3 行 4 列。
- 返回对象的类型为 `torch.Tensor` ，它是 `torch.FloatTensor` 的别名；默认情况下，PyTorch 张量使用 32 位浮点数填充。（更多数据类型信息请参见下文。）
- 打印张量时，你可能会看到一些看似随机的值。`torch.empty `torch.empty()` ` 调用会为张量分配内存，但不会用任何值初始化它——所以你看到的是分配内存时内存中的所有内容。

关于张量及其维度和术语的简要说明：

- 你有时会看到一个名为 1 维张量的张量 *向量。*
- 同样，二维张量通常被称为 *矩阵。*
- 任何维度超过两个维度的量通常都称为张量。

通常情况下，你需要用一些值来初始化你的张量。 值。常见情况包括全零、全一或随机值，以及 `torch` 模块为所有这些操作提供了工厂方法：

```
zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)
```

```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
```
如你所料，我们有一个全为零的张量，一个全为一的张量，以及一个介于 0 和 1 之间的随机值的张量。

### 随机张量和种子#

说到随机张量，你注意到对……的调用了吗？ 在 `torch.manual_seed()` 之前紧接着执行什么操作？用随机值初始化张量（例如模型的学习权重）很常见，但有时——尤其是在研究环境中——你需要确保结果的可复现性。手动设置随机数生成器的种子是实现这一目标的方法。让我们仔细看看：

```
torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)
```



```
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
tensor([[0.3126, 0.3791, 0.3087],
        [0.0736, 0.4216, 0.0691]])
tensor([[0.2332, 0.4047, 0.2162],
        [0.9927, 0.4128, 0.5938]])
```



如上所示， `random1` 和 `random3` 值相同， `random2` 和 `random4` 值也相同。手动设置随机数生成器的种子会重置它，因此在大多数情况下，依赖于随机数的相同计算应该会得到相同的结果。

有关更多信息，请参阅 [PyTorch 文档中的可复现性部分 ](https://pytorch.org/docs/stable/notes/randomness.html)。

### 张量形状#

通常，当对两个或多个张量执行操作时，它们需要具有相同的 *形状* ——也就是说，具有相同的维度数以及每个维度上相同数量的单元格。为此，我们提供了 `torch.*_like()` 方法：

```
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)
```



```
torch.Size([2, 2, 3])
tensor([[[-1.2804e-08,  3.0757e-41,  1.4013e-45],
         [ 0.0000e+00,  1.4013e-45,  0.0000e+00]],

        [[ 1.4013e-45,  0.0000e+00,  1.4013e-45],
         [ 0.0000e+00,  1.4013e-45,  0.0000e+00]]])
torch.Size([2, 2, 3])
tensor([[[-2.7227e+25,  3.0750e-41, -2.8468e+25],
         [ 3.0750e-41,  0.0000e+00,  1.8750e+00]],

        [[ 0.0000e+00,  1.8750e+00,  0.0000e+00],
         [ 1.8750e+00,  0.0000e+00,  1.8750e+00]]])
torch.Size([2, 2, 3])
tensor([[[0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.]]])
torch.Size([2, 2, 3])
tensor([[[1., 1., 1.],
         [1., 1., 1.]],

        [[1., 1., 1.],
         [1., 1., 1.]]])
torch.Size([2, 2, 3])
tensor([[[0.6128, 0.1519, 0.0453],
         [0.5035, 0.9978, 0.3884]],

        [[0.6929, 0.1703, 0.1384],
         [0.4759, 0.7481, 0.0361]]])
```



上面代码单元格中的第一个新内容是使用了 `.shape` 张量的属性。此属性包含一个范围列表。 张量的每个维度——在我们的例子中， `x` 是一个形状为 2 x 2 x 3 的三维张量。

下面，我们调用 `.empty_like()` 、. `.zeros_like()` 、 `.ones_like()` 和 `.rand_like()` 方法。使用 `.shape` 通过属性，我们可以验证这些方法都返回一个张量。 维度和范围完全相同。

最后一种创建能够覆盖整个数据集的张量的方法是直接从 PyTorch 集合中指定其数据：

```
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)
```



```
tensor([[3.1416, 2.7183],
        [1.6180, 0.0073]])
tensor([ 2,  3,  5,  7, 11, 13, 17, 19])
tensor([[2, 4, 6],
        [3, 6, 9]])
```



如果您已经拥有以 Python 元组或列表形式存储的数据，那么使用 `torch.tensor()` 是创建张量的最直接方法。如上所示，嵌套这些集合将生成一个多维张量。

笔记

`torch.tensor()` 创建数据副本。

### 张量数据类型#

设置张量的数据类型有两种方法：

```
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)
```



```
tensor([[1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
tensor([[ 0.9956,  1.4148,  5.8364],
        [11.2406, 11.2083, 11.6692]], dtype=torch.float64)
tensor([[ 0,  1,  5],
        [11, 11, 11]], dtype=torch.int32)
```



设置张量底层数据类型的最简单方法是在创建时使用可选参数。在上面单元格的第一行中，我们为张量 `a` 设置了 `dtype=torch.int16` 。当我们打印 `a` 时，可以看到它全是 `1` ，而不是 `1.` 这是 Python 的一个微妙提示，表明这是一个整数类型而不是浮点数类型。

打印 `a` 量时需要注意的另一点是，与将 `dtype` 保留为默认值（32 位浮点数）不同，打印张量时还会指定其 `dtype` 。

您可能也注意到，我们之前将张量的形状指定为一系列整数参数，现在改为将这些参数分组到一个元组中。这并非绝对必要——PyTorch 本身就接受一系列初始的、未标记的整数参数作为张量形状——但添加可选参数后，可以使您的代码更易读。

设置数据类型的另一种方法是使用 `.to()` 方法。在上面的单元格中，我们以常规方式创建了一个随机浮点张量 `b` 。然后，我们使用 `.to()` 方法将 `b` 转换为 32 位整数，从而创建 `c` 。请注意， `c` 包含与 `b` 相同的所有值，但都被截断为整数。

有关更多信息，请参阅[数据类型文档 ](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)。

## 使用 PyTorch 张量进行数学和逻辑运算#

既然你已经了解了一些创建张量的方法……那么你可以用它们做什么呢？

我们先来看一些基本的算术运算，以及张量如何与简单的标量相互作用：

```
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)
```



```
tensor([[1., 1.],
        [1., 1.]])
tensor([[2., 2.],
        [2., 2.]])
tensor([[3., 3.],
        [3., 3.]])
tensor([[4., 4.],
        [4., 4.]])
tensor([[1.4142, 1.4142],
        [1.4142, 1.4142]])
```



如上所示，张量和标量之间的算术运算，例如加法、减法、乘法、除法和乘方，是分布在张量的每个元素上的。由于此类运算的输出将是一个张量，因此您可以按照通常的运算符优先级规则将它们链接在一起，就像我们创建 `threes` 那一行一样。

两个张量之间的类似运算也符合你的直觉预期：

```
powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)
```



```
tensor([[ 2.,  4.],
        [ 8., 16.]])
tensor([[5., 5.],
        [5., 5.]])
tensor([[12., 12.],
        [12., 12.]])
```



需要注意的是，前一个代码单元中的所有张量形状都相同。如果我们尝试对形状不同的张量执行二元运算，会发生什么情况？

笔记

以下单元格会抛出运行时错误。这是有意为之。

```
a = torch.rand(2, 3)
b = torch.rand(3, 2)

print(a * b)
```



一般情况下，即使像上面单元格中张量具有相同数量元素的情况，也不能以这种方式对不同形状的张量进行操作。

### 简述：张量广播#

笔记

如果您熟悉 NumPy ndarray 中的广播语义，您会发现同样的规则也适用于这里。

相同形状规则的例外情况是*张量广播。* 以下是一个示例：

```
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)
```



```
tensor([[0.6146, 0.5999, 0.5013, 0.9397],
        [0.8656, 0.5207, 0.6865, 0.3614]])
tensor([[1.2291, 1.1998, 1.0026, 1.8793],
        [1.7312, 1.0413, 1.3730, 0.7228]])
```



这里有什么诀窍？为什么我们要将一个 2x4 的张量乘以一个 1x4 的张量？

广播是一种对形状相似的张量进行运算的方法。在上面的例子中，单行四列的张量与两行四列的张量的 *两行*相乘。

这是深度学习中的一项重要操作。常见的例子是将学习权重张量与 *一批*输入张量相乘，分别对批次中的每个实例应用该操作，并返回一个形状相同的张量——就像我们上面的 (2, 4) * (1, 4) 示例返回了一个形状为 (2, 4) 的张量一样。

广播规则如下：

- 每个张量必须至少有一个维度——不能有空张量。
- 比较两个张量的维度大小， *从后到前：*
  - 每个维度必须相等， *或者*
  - 其中一个尺寸必须为 1， *或者*
  - 该维度在其中一个张量中不存在

当然，形状相同的张量很容易“广播”，正如你之前看到的那样。

以下是一些符合上述规则并允许广播的情况示例：

```
a =     torch.ones(4, 3, 2)

b = a * torch.rand(   3, 2) # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(   1, 2) # 3rd dim identical to a, 2nd dim = 1
print(d)
```



```
tensor([[[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]],

        [[0.6493, 0.2633],
         [0.4762, 0.0548],
         [0.2024, 0.5731]]])
tensor([[[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]],

        [[0.7191, 0.7191],
         [0.4067, 0.4067],
         [0.7301, 0.7301]]])
tensor([[[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]],

        [[0.6276, 0.7357],
         [0.6276, 0.7357],
         [0.6276, 0.7357]]])
```



仔细观察上面每个张量的值：

- 创建 `b` 乘法运算被广播到 `a` 的每一“层”。
- 对于 `c` ，该操作被广播到每一层和每一行。 `a` - 每列三个元素都是相同的。
- 对于 `d` ，我们进行了调整——现在每一*行* ，无论跨层还是跨列，都完全相同。

有关广播的更多信息，请参阅 [PyTorch 文档。 ](https://pytorch.org/docs/stable/notes/broadcasting.html)关于这个话题。

以下是一些广播尝试失败的例子：

笔记

以下单元格会抛出运行时错误。这是有意为之。

```
a =     torch.ones(4, 3, 2)

b = a * torch.rand(4, 3)    # dimensions must match last-to-first

c = a * torch.rand(   2, 3) # both 3rd & 2nd dims different

d = a * torch.rand((0, ))   # can't broadcast with an empty tensor
```



### 张量数学详解#

PyTorch 张量可以执行超过三百种操作。

以下是一些主要业务类别的简要示例：

```
# common functions
a = torch.rand(2, 4) * 2 - 1
print('Common functions:')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))

# trigonometric functions and their inverses
angles = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
sines = torch.sin(angles)
inverses = torch.asin(sines)
print('\nSine and arcsine:')
print(angles)
print(sines)
print(inverses)

# bitwise operations
print('\nBitwise XOR:')
b = torch.tensor([1, 5, 11])
c = torch.tensor([2, 7, 10])
print(torch.bitwise_xor(b, c))

# comparisons:
print('\nBroadcasted, element-wise equality comparison:')
d = torch.tensor([[1., 2.], [3., 4.]])
e = torch.ones(1, 2)  # many comparison ops support broadcasting!
print(torch.eq(d, e)) # returns a tensor of type bool

# reductions:
print('\nReduction ops:')
print(torch.max(d))        # returns a single-element tensor
print(torch.max(d).item()) # extracts the value from the returned tensor
print(torch.mean(d))       # average
print(torch.std(d))        # standard deviation
print(torch.prod(d))       # product of all numbers
print(torch.unique(torch.tensor([1, 2, 1, 2, 1, 2]))) # filter unique elements

# vector and linear algebra operations
v1 = torch.tensor([1., 0., 0.])         # x unit vector
v2 = torch.tensor([0., 1., 0.])         # y unit vector
m1 = torch.rand(2, 2)                   # random matrix
m2 = torch.tensor([[3., 0.], [0., 3.]]) # three times identity matrix

print('\nVectors & Matrices:')
print(torch.linalg.cross(v2, v1)) # negative of z unit vector (v1 x v2 == -v2 x v1)
print(m1)
m3 = torch.linalg.matmul(m1, m2)
print(m3)                  # 3 times m1
print(torch.linalg.svd(m3))       # singular value decomposition
```



```
Common functions:
tensor([[0.9238, 0.5724, 0.0791, 0.2629],
        [0.1986, 0.4439, 0.6434, 0.4776]])
tensor([[-0., -0., 1., -0.],
        [-0., 1., 1., -0.]])
tensor([[-1., -1.,  0., -1.],
        [-1.,  0.,  0., -1.]])
tensor([[-0.5000, -0.5000,  0.0791, -0.2629],
        [-0.1986,  0.4439,  0.5000, -0.4776]])

Sine and arcsine:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 0.7854])

Bitwise XOR:
tensor([3, 2, 1])

Broadcasted, element-wise equality comparison:
tensor([[ True, False],
        [False, False]])

Reduction ops:
tensor(4.)
4.0
tensor(2.5000)
tensor(1.2910)
tensor(24.)
tensor([1, 2])

Vectors & Matrices:
tensor([ 0.,  0., -1.])
tensor([[0.7375, 0.8328],
        [0.8444, 0.2941]])
tensor([[2.2125, 2.4985],
        [2.5332, 0.8822]])
torch.return_types.linalg_svd(
U=tensor([[-0.7889, -0.6145],
        [-0.6145,  0.7889]]),
S=tensor([4.1498, 1.0548]),
Vh=tensor([[-0.7957, -0.6056],
        [ 0.6056, -0.7957]]))
```



这只是部分操作示例。如需了解更多详情和完整清单，请联系我们。 数学函数，请查看 [文档 ](https://pytorch.org/docs/stable/torch.html#math-operations)。有关更多详细信息和线性代数运算的完整列表，请参阅此 [文档 ](https://pytorch.org/docs/stable/linalg.html)。

### 原地修改张量#

大多数对张量的二元运算都会返回第三个新的张量。当我们说 `c = a * b` （其中 `a` 和 `b` 是张量）时，新的张量 `c` 将占据与其他张量不同的内存区域。

不过，有时您可能需要直接修改张量——例如，当您进行逐元素计算并需要舍弃中间值时。为此，大多数数学函数都提供了一个带有下划线 ( `_` ) 的版本，用于直接修改张量。

例如：

```
a = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('a:')
print(a)
print(torch.sin(a))   # this operation creates a new tensor in memory
print(a)              # a has not changed

b = torch.tensor([0, math.pi / 4, math.pi / 2, 3 * math.pi / 4])
print('\nb:')
print(b)
print(torch.sin_(b))  # note the underscore
print(b)              # b has changed
```



```
a:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7854, 1.5708, 2.3562])

b:
tensor([0.0000, 0.7854, 1.5708, 2.3562])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
tensor([0.0000, 0.7071, 1.0000, 0.7071])
```



对于算术运算，存在一些行为类似的函数：

```
a = torch.ones(2, 2)
b = torch.rand(2, 2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b)
```



```
Before:
tensor([[1., 1.],
        [1., 1.]])
tensor([[0.3788, 0.4567],
        [0.0649, 0.6677]])

After adding:
tensor([[1.3788, 1.4567],
        [1.0649, 1.6677]])
tensor([[1.3788, 1.4567],
        [1.0649, 1.6677]])
tensor([[0.3788, 0.4567],
        [0.0649, 0.6677]])

After multiplying
tensor([[0.1435, 0.2086],
        [0.0042, 0.4459]])
tensor([[0.1435, 0.2086],
        [0.0042, 0.4459]])
```



请注意，这些原地算术函数是方法。 `torch.Tensor` 对象，不像许多其他函数（例如 `torch.sin()` ）那样附加到 `torch` 模块。正如您从此处看到的 `a.add_(b)` ， *调用张量会被原地改变。*

还有另一种方法可以将计算结果放入已分配的现有张量中。我们目前看到的许多方法和函数（包括创建方法！）都有一个 `out` 参数，允许您指定一个张量来接收输出。如果 `out` 张量的形状和 `dtype` 正确，则无需分配新的内存即可完成此操作：

```
a = torch.rand(2, 2)
b = torch.rand(2, 2)
c = torch.zeros(2, 2)
old_id = id(c)

print(c)
d = torch.matmul(a, b, out=c)
print(c)                # contents of c have changed

assert c is d           # test c & d are same object, not just containing equal values
assert id(c) == old_id  # make sure that our new c is the same object as the old one

torch.rand(2, 2, out=c) # works for creation too!
print(c)                # c has changed again
assert id(c) == old_id  # still the same object!
```



```
tensor([[0., 0.],
        [0., 0.]])
tensor([[0.3653, 0.8699],
        [0.2364, 0.3604]])
tensor([[0.0776, 0.4004],
        [0.9877, 0.0352]])
```



## 复制张量#

与 Python 中的任何对象一样，将张量赋值给变量只会使该变量成为张量的 *标签* ，而不会复制张量本身。例如：

```
a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)       # ...and b is also altered
```



```
tensor([[  1., 561.],
        [  1.,   1.]])
```



但如果你想要一份单独的数据副本进行操作呢？ `clone()` 方法可供您使用：

```
a = torch.ones(2, 2)
b = a.clone()

assert b is not a      # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561          # a changes...
print(b)               # ...but b is still all ones
```



```
tensor([[True, True],
        [True, True]])
tensor([[1., 1.],
        [1., 1.]])
```



**使用 ``clone()`` 时，有一件重要的事情需要注意。** 如果源张量启用了自动微分，那么克隆张量也将启用自动微分。 **这部分内容将在关于自动微分的视频中更深入地讲解，** 但如果您只想了解简要细节，请继续阅读。

*在许多情况下，这正是你想要的。* 例如，如果你的模型在其 `forward()` 方法中有多个计算路径，并且*两者* 原始张量及其克隆都会对模型的输出产生贡献。 要启用模型学习，需要为两个张量都启用自动微分。 如果你的源张量启用了自动微分（通常情况下都会启用）， 它是一组学习权重，或是通过涉及以下方面的计算得出的： 如果调整权重），你就能得到想要的结果。

另一方面，如果你进行的计算*既不*需要原始张量也不需要其克隆跟踪梯度，那么只要源张量的自动微分功能关闭，就可以正常进行计算。

*还有第三种情况* ：假设你在模型的 `forward()` 函数中执行计算，默认情况下所有计算都启用了梯度跟踪，但你想在计算过程中提取一些值来生成一些指标。在这种情况下，你 *不*希望源张量的克隆副本跟踪梯度——关闭自动微分的历史跟踪功能可以提高性能。为此，你可以对源张量使用 `.detach()` 方法：

```
a = torch.rand(2, 2, requires_grad=True) # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)
```



```
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], grad_fn=<CloneBackward0>)
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]])
tensor([[0.0905, 0.4485],
        [0.8740, 0.2526]], requires_grad=True)
```



这里发生了什么事？

- 我们创建了 `a` 启用了 `requires_grad=True` 实例。 **我们尚未介绍这个可选参数，但会在自动微分单元中进行讲解。**
- 当我们打印 `a` 时，它会告诉我们该属性 `requires_grad=True` - 这意味着自动微分和计算历史跟踪已启用。
- 我们克隆 `a` 并将其标记为 `b` 。当我们打印 `b` 时，我们可以看到 它正在跟踪其计算历史——它继承了 `a` 的自动微分设置，并添加到计算历史记录中。
- 我们将 `a` 克隆到 `c` 中，但我们首先调用 `detach()` 。
- 打印 `c` ，我们看不到任何计算历史记录，也没有 `requires_grad=True` 。

` `detach()` 方法*将张量与其计算历史记录分离。* 它的意思是：“执行接下来的所有操作，就好像自动微分功能已关闭一样。” 这样做 *不会*改变 `a` ——你可以从打印结果中看到这一点。 `a` ，它仍然保留 `requires_grad=True` 属性。

## 迈向[加速器 ](https://pytorch.org/docs/stable/torch.html#accelerators)

PyTorch 的主要优势之一是其在英伟达平台上的强大加速能力。 [加速器](https://pytorch.org/docs/stable/torch.html#accelerators) 例如 CUDA、MPS、MTIA 或 XPU。 到目前为止，我们所有的工作都是在 CPU 上进行的。我们如何才能过渡到更快的处理器呢？ 硬件？

首先，我们应该检查是否有加速器可用，方法是： `is_available()` 方法。

笔记

如果没有加速器，则本节中的可执行单元将不会执行任何与加速器相关的代码。

```
if torch.accelerator.is_available():
    print('We have an accelerator!')
else:
    print('Sorry, CPU only.')
```


```
We have an accelerator!
```


一旦确定有一个或多个加速器可用，我们就需要将数据放在加速器可以访问的位置。CPU 在计算机的 RAM 中对数据进行计算。加速器则配备了专用内存。每当需要在某个设备上执行计算时，都必须将计算所需的 *所有*数据移动到该设备可访问的内存中。（通俗地说，“将数据移动到 GPU 可访问的内存”简称为“将数据移动到 GPU”。）

将数据传输到目标设备的方法有很多种。您可以在创建数据时进行传输：

```
if torch.accelerator.is_available():
    gpu_rand = torch.rand(2, 2, device=torch.accelerator.current_accelerator())
    print(gpu_rand)
else:
    print('Sorry, CPU only.')
```



```
tensor([[0.3344, 0.2640],
        [0.2119, 0.0582]], device='cuda:0')
```



默认情况下，新张量是在 CPU 上创建的，因此我们需要指定 当我们想在加速器上创建张量时，可以选择使用以下参数： `device` 参数。可以看到，当我们打印新张量时，PyTorch 会告诉我们它位于哪个设备上（如果它不在 CPU 上）。

您可以使用 `torch.accelerator.device_count()` 查询加速器的数量。 如果您有多个加速器，可以通过索引指定它们，例如 CUDA： `device='cuda:0'` ， `device='cuda:1'` ，等等。

作为一种编码实践，到处使用字符串常量来指定设备是非常脆弱的。理想情况下，无论是在 CPU 还是加速器硬件上，你的代码都应该能够稳定运行。你可以通过创建一个设备句柄，并将其传递给张量而不是字符串来实现这一点：

```
my_device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
```



```
Device: cuda
tensor([[0.0024, 0.6778],
        [0.2441, 0.6812]], device='cuda:0')
```



如果一个张量已经存在于一个设备上，可以使用 `to()` 方法将其移动到另一个设备。以下代码行在 CPU 上创建一个张量，并将其移动到你在前一个单元格中获取的设备句柄。

```
y = torch.rand(2, 2)
y = y.to(my_device)
```



需要注意的是，要进行涉及两个或多个张量的计算， *所有张量必须位于同一设备上* 。以下代码会抛出运行时错误，无论是否有加速设备可用，例如 CUDA：

```
x = torch.rand(2, 2)
y = torch.rand(2, 2, device='cuda')
z = x + y  # exception will be thrown
```



## 操控张量形状#

有时，你需要改变张量的形状。下面，我们将介绍几种常见情况以及相应的处理方法。

### 更改维度数量#

当你向模型传递单个输入实例时，可能需要更改维度数量。PyTorch 模型通常需要 *批量*输入。

例如，假设你有一个模型，它处理的是 3 x 226 x 226 的图像——一个边长为 226 像素、具有 3 个颜色通道的正方形。当你加载并转换它时，你会得到一个形状为 `(3, 226, 226)` 的张量。然而，你的模型期望的输入形状是 `(N, 3, 226, 226)` ，其中 `N` 是批次中图像的数量。那么，如何创建一个包含一张图像的批次呢？

```
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)
```



```
torch.Size([3, 226, 226])
torch.Size([1, 3, 226, 226])
```



`unsqueeze()` 方法增加了一个范围为 1 的维度。 `unsqueeze(0)` 将其添加为新的零维 - 现在你有一批数量为 1 的物体！

如果这就是*所谓的“反*挤压”呢？那么“挤压”又是什么意思呢？我们利用的是这样一个事实：任何尺度为 1 的维度 *都不会*改变张量中的元素数量。

```
c = torch.rand(1, 1, 1, 1, 1)
print(c)
```



```
tensor([[[[[0.2347]]]]])
```



继续上面的例子，假设模型的输出对于每个输入都是一个包含 20 个元素的向量。那么你会期望输出的形状为 `(N, 20)` ，其中 `N` 是输入批次中的实例数。这意味着对于我们的单输入批次，我们将得到形状为 `(1, 20)` 的输出。

如果要对该输出进行一些*非批量*计算（例如，需要一个包含 20 个元素的向量），该怎么办？

```
a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)
```



```
torch.Size([1, 20])
tensor([[0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
         0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
         0.2792, 0.3277]])
torch.Size([20])
tensor([0.1899, 0.4067, 0.1519, 0.1506, 0.9585, 0.7756, 0.8973, 0.4929, 0.2367,
        0.8194, 0.4509, 0.2690, 0.8381, 0.8207, 0.6818, 0.5057, 0.9335, 0.9769,
        0.2792, 0.3277])
torch.Size([2, 2])
torch.Size([2, 2])
```



从形状可以看出，我们的二维张量现在变成了一维张量，如果你仔细观察上面单元格的输出，你会发现打印 `a` 会显示一组“额外的”方括号。 `[]` 因为多了一个维度。

你只能对范围为 1 的维度 `squeeze()` 操作。参见上文，我们尝试对 `c` 中大小为 2 的维度进行压缩，结果恢复了初始形状。squeeze `squeeze()` 和 `unsqueeze()` 函数只能作用于范围为 1 的维度，否则会改变张量中的元素数量。

`unsqueeze()` 另一个用途是简化广播。回想一下上面的例子，我们有以下代码：

```
a = torch.ones(4, 3, 2)

c = a * torch.rand(   3, 1) # 3rd dim = 1, 2nd dim identical to a
print(c)
```



这样做的最终结果是将操作广播到维度 0 和 2，导致随机的 3 x 1 张量逐元素地与 `a` 中的每个 3 元素列相乘。

如果随机向量只是一个包含 3 个元素的向量呢？我们将无法进行广播，因为最终的维度将不符合广播规则。`unsqueeze `unsqueeze()` 可以解决这个问题：

```
a = torch.ones(4, 3, 2)
b = torch.rand(   3)     # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)       # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)             # broadcasting works again!
```



```
torch.Size([3, 1])
tensor([[[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]],

        [[0.1891, 0.1891],
         [0.3952, 0.3952],
         [0.9176, 0.9176]]])
```



`squeeze()` 和 `unsqueeze()` 方法也有原地版本，分别是 `squeeze_()` 和 `unsqueeze_()` ：

```
batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)
```



```
torch.Size([3, 226, 226])
torch.Size([1, 3, 226, 226])
```



有时，您可能需要更彻底地改变张量的形状，同时还要保持元素数量及其内容不变。这种情况常见于模型的卷积层和线性层之间的接口处——这在图像分类模型中尤为常见。卷积核会生成一个形状为 *features x width x height 的输出张量，* 但后续的线性层需要一个一维输入。`reshape `reshape()` 可以帮您实现这一点，前提是您请求的维度与输入张量的元素数量相同：

```
output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)
```



```
torch.Size([6, 20, 20])
torch.Size([2400])
torch.Size([2400])
```


上面单元格最后一行的参数 `(6 * 20 * 20,)` 是因为 PyTorch 在指定张量形状时期望的是一个 **元组** ——但当形状是方法的第一个参数时，它允许我们“作弊”，直接使用一系列整数。这里，我们必须添加括号和逗号，才能让方法相信这确实是一个包含一个元素的元组。

如果条件允许， `reshape()` 会返回待修改张量的*视图* ——也就是说，返回一个指向同一底层内存区域的独立张量对象。 *这一点很重要：* 这意味着对源张量所做的任何更改都会反映在该张量的视图中，除非你 `clone()` 函数克隆它。

*还有*一些超出本介绍范围的情况， `reshape()` 必须返回一个包含数据副本的张量。 更多信息，请参见 [文档 ](https://pytorch.org/docs/stable/torch.html#torch.reshape)。

## NumPy Bridge

在上面关于广播的部分中提到，PyTorch 的广播语义与 NumPy 的广播语义兼容——但 PyTorch 和 NumPy 之间的联系远不止于此。

如果您已有机器学习或科学计算代码，且数据存储在 NumPy ndarray 中，您可能希望将相同的数据表示为 PyTorch 张量，以便利用 PyTorch 的 GPU 加速或其高效的机器学习模型构建抽象。在 ndarray 和 PyTorch 张量之间切换非常简单：

```
import numpy as np

numpy_array = np.ones((2, 3))
print(numpy_array)

pytorch_tensor = torch.from_numpy(numpy_array)
print(pytorch_tensor)
```



```
[[1. 1. 1.]
 [1. 1. 1.]]
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
```



PyTorch 创建一个与 NumPy 数组形状相同、包含相同数据的张量，甚至保留了 NumPy 的默认 64 位浮点数据类型。

这种转换同样可以很容易地反向进行：

```
pytorch_rand = torch.rand(2, 3)
print(pytorch_rand)

numpy_rand = pytorch_rand.numpy()
print(numpy_rand)
```

```
tensor([[0.8716, 0.2459, 0.3499],
        [0.2853, 0.9091, 0.5695]])
[[0.87163675 0.2458961  0.34993553]
 [0.2853077  0.90905803 0.5695162 ]]
```

需要注意的是，这些转换后的对象与它们的源对象使用*相同的底层内存* ，这意味着对其中一个对象的更改会反映在另一个对象中：

```
numpy_array[1, 1] = 23
print(pytorch_tensor)

pytorch_rand[1, 1] = 17
print(numpy_rand)
```

```
tensor([[ 1.,  1.,  1.],
        [ 1., 23.,  1.]], dtype=torch.float64)
[[ 0.87163675  0.2458961   0.34993553]
 [ 0.2853077  17.          0.5695162 ]]
```

