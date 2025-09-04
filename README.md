# Jittor超声图像的智能筛查与分级任务

## 简介

本项目包含了第五届计图挑战赛赛道一超声图像的智能筛查与分级任务的代码实现。该任务面临类别分布极度不均衡与类别间特征相似度高等挑战，要求模型具有较强的泛化能力和对少数类的敏感性。

### 解决思路

为应对上述挑战，本项目从损失函数、数据增强、采样策略等多个方面设计了综合方案：

- 主干网络：采用 Swin Transformer（`swin_base_patch4_window12_384`）作为图像主干，结合其在视觉任务中的局部建模与全局感知能力；
- 损失函数：引入 RW-LDAM-DRW Loss，融合类别敏感 margin、类别频次重加权与延迟启用机制，以增强对尾类样本的判别能力；
- 数据增强：基于 Albumentations 实现多种图像增强策略（翻转、旋转、遮挡等），提升模型鲁棒性；
- 类别平衡：结合手动设定的类别过采样比例，有效缓解数据不平衡问题；
- 难例挖掘：使用 Online Hard Example Mining（OHEM）策略，强化对易混淆样本的关注；
- 模型稳定性：引入 Exponential Moving Average（EMA）机制，平滑模型参数，提升推理稳定性；
- 评估方法：采用 K 折交叉验证，结合多模型融合策略，提高最终模型性能评估的可靠性。

## 安装 

本项目可在一张 NVIDIA 4090 GPU 上运行，预计训练时间为4小时。

#### 运行环境
- ubuntu 22.04.5 LTS
- python >= 3.7
- jittor >= 1.3.10.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型
本项目使用swin_base_patch4_window12_384模型。

## 训练

训练可运行以下命令：
```
bash Mycode1/run_checkpoint.sh
```

## 推理

如果只需要推理可以尝试以下命令：

```
bash Mycode1/run_checkpoint_testonly.sh
```

## 致谢

此项目的损失函数基于论文 *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* 实现，部分代码参考了 [Jittor-Image-Models](https://github.com/Jittor-Image-Models/Jittor-Image-Models)。

