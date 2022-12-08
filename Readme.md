## 目的

寻找一种训练算法，能够直接提升分割模型对运动模糊图像的检测能力

从模型初始化、模型特征蒸馏算法、模型标签蒸馏算法进行提升

## 步骤

我想说明的，这种方法能够提升检测能力：

### 验证

1、这种检测能力与数据集有关（这个只是做验证用的）

2、与初始化的模型类型有关

### 比较

3、与去模糊模型的性能有关

4、与初始化模型权重的比例有关

5、标签蒸馏算法与特征蒸馏算法

首先设置清晰图像的分割训练超参基准

确定Batch_Size、LearningRate、LeaeningRate Schedule、ImageAugmentation、ImageSize、Epochs

目前跑了Batch_Size 为4、8 希望跑一下 2、16、32 的结果

选择最好的作为基准

然后讨论模糊模型

选择ResNet，ConvNext，FPNInception，MobileNetv3

讨论初始化

讨论数据集

讨论蒸馏算法
