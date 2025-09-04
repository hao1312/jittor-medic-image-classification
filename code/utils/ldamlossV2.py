import jittor as jt
import jittor.nn as nn
import numpy as np
import math

class LDAMLoss(nn.Module):
    """
    LDAM (Label-Distribution-Aware Margin) Loss for Jittor
    
    Args:
        cls_num_list: 每个类别的样本数量列表
        max_m: 最大边距值，默认0.5
        weight: 类别权重，用于重加权
        s: 缩放因子，类似于余弦损失中的温度参数，默认30
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        
        # 计算每个类别的边距
        # 边距与类别样本数的四次方根成反比
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        
        # 转换为Jittor张量
        self.m_list = jt.array(m_list).float()
        self.s = s
        self.weight = weight
        
    def execute(self, input, target, reduction='mean'):
        """
        Args:
            input: 模型输出的logits，形状为 [batch_size, num_classes]
            target: 真实标签，形状为 [batch_size]
            reduction: 'mean', 'sum', 'none' - 控制损失聚合方式
        """
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        
        # 创建one-hot编码
        index = jt.zeros([batch_size, num_classes])
        target_long = target.long()
        for i in range(batch_size):
            index[i, target_long[i]] = 1.0
        
        # 为每个样本应用对应类别的边距
        # 获取每个样本对应类别的边距值
        margins = self.m_list[target_long]  # [batch_size]
        margins = margins.unsqueeze(1).expand(batch_size, num_classes)  # [batch_size, num_classes]
        
        # 只对真实类别应用边距
        x_m = input - margins * index
        
        # 应用缩放因子并计算交叉熵损失
        output = jt.where(index.bool(), x_m, input)
        # 使用函数版本的cross_entropy_loss，支持reduction参数
        return nn.cross_entropy_loss(self.s * output, target, weight=self.weight, reduction=reduction)
    
    def get_per_sample_loss(self, input, target):
        """
        获取每个样本的独立损失值
        
        Returns:
            per_sample_losses: 形状为 [batch_size] 的张量，包含每个样本的损失
        """
        return self.execute(input, target, reduction='none')


def get_class_weights(cls_num_list, reweight_type='inverse'):
    """
    计算类别权重用于重加权
    
    Args:
        cls_num_list: 每个类别的样本数量
        reweight_type: 重加权类型，'inverse' 或 'sqrt_inverse'
    """
    cls_num_list = np.array(cls_num_list)
    
    if reweight_type == 'inverse':
        per_cls_weights = 1.0 / cls_num_list
    elif reweight_type == 'sqrt_inverse':
        per_cls_weights = 1.0 / np.sqrt(cls_num_list)
    else:
        raise ValueError(f"Unknown reweight_type: {reweight_type}")
    
    # 归一化权重
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    
    return jt.array(per_cls_weights).float()


class LDAMDRWLoss(nn.Module):
    """
    LDAM-DRW 损失的完整实现
    结合了LDAM损失和延迟重加权策略
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30, reweight_epoch=160, 
                 total_epochs=200, reweight_type='inverse'):
        super(LDAMDRWLoss, self).__init__()
        
        self.cls_num_list = cls_num_list
        self.num_classes = len(cls_num_list)
        self.reweight_epoch = reweight_epoch
        self.total_epochs = total_epochs
        
        # 初始化不带权重的LDAM损失
        self.ldam_loss = LDAMLoss(cls_num_list, max_m=max_m, s=s)
        
        # 计算重加权权重
        self.reweight_weights = get_class_weights(cls_num_list, reweight_type)
        
        # 初始化带权重的LDAM损失（用于DRW阶段）
        self.ldam_loss_reweight = LDAMLoss(cls_num_list, max_m=max_m, s=s, 
                                          weight=self.reweight_weights)
    
    def execute(self, input, target, epoch, reduction='mean'):
        """
        根据当前epoch决定使用哪种损失
        
        Args:
            input: 模型输出
            target: 真实标签  
            epoch: 当前训练轮次
            reduction: 'mean', 'sum', 'none' - 控制损失聚合方式
        """
        if epoch < self.reweight_epoch:
            # 第一阶段：使用普通LDAM损失
            return self.ldam_loss(input, target, reduction=reduction)
        else:
            # 第二阶段：使用重加权LDAM损失
            return self.ldam_loss_reweight(input, target, reduction=reduction)
    
    def get_per_sample_loss(self, input, target, epoch):
        """
        获取每个样本的独立损失值，用于OHEM
        
        Returns:
            per_sample_losses: 形状为 [batch_size] 的张量，包含每个样本的损失
        """
        return self.execute(input, target, epoch, reduction='none')

class RWLDAMDRWLoss(nn.Module):
    """
    RW LDAM-DRW (ReWeighted LDAM-DRW) 损失
    这是标准的LDAM-DRW实现，使用重加权策略
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30, reweight_epoch=160, 
                 total_epochs=200, reweight_type='inverse'):
        super(RWLDAMDRWLoss, self).__init__()
        
        self.cls_num_list = cls_num_list
        self.num_classes = len(cls_num_list)
        self.reweight_epoch = reweight_epoch
        self.total_epochs = total_epochs
        
        # 初始化不带权重的LDAM损失
        self.ldam_loss = LDAMLoss(cls_num_list, max_m=max_m, s=s)
        
        # 计算重加权权重
        self.reweight_weights = get_class_weights(cls_num_list, reweight_type)
        
        # 初始化带权重的LDAM损失（用于DRW阶段）
        self.ldam_loss_reweight = LDAMLoss(cls_num_list, max_m=max_m, s=s, 
                                          weight=self.reweight_weights)
    
    def execute(self, input, target, epoch, reduction='mean'):
        """
        根据当前epoch决定使用哪种损失
        
        Args:
            input: 模型输出
            target: 真实标签  
            epoch: 当前训练轮次
            reduction: 'mean', 'sum', 'none' - 控制损失聚合方式
        """
        if epoch < self.reweight_epoch:
            # 第一阶段：使用普通LDAM损失
            return self.ldam_loss(input, target, reduction=reduction)
        else:
            # 第二阶段：使用重加权LDAM损失
            return self.ldam_loss_reweight(input, target, reduction=reduction)
    
    def get_per_sample_loss(self, input, target, epoch):
        """
        获取每个样本的独立损失值，用于OHEM
        
        Returns:
            per_sample_losses: 形状为 [batch_size] 的张量，包含每个样本的损失
        """
        return self.execute(input, target, epoch, reduction='none')
    
    def get_uncertainty_scores(self, input, target, epoch):
        """
        获取每个样本的不确定性度量，可用于样本选择
        
        Args:
            input: 模型输出的logits，形状为 [batch_size, num_classes]
            target: 真实标签，形状为 [batch_size]
            epoch: 当前训练轮次
        
        Returns:
            uncertainty_scores: 形状为 [batch_size] 的张量，包含每个样本的不确定性分数
                              分数越高表示样本越难/越不确定
        """
        # 方法1: 使用损失值作为不确定性度量（损失越高越不确定）
        per_sample_losses = self.get_per_sample_loss(input, target, epoch)
        
        # 方法2: 使用预测概率的熵作为不确定性度量
        with jt.no_grad():
            probs = jt.nn.softmax(input, dim=1)
            entropy = -(probs * jt.log(probs + 1e-8)).sum(dim=1)
        
        # 方法3: 使用预测置信度的倒数作为不确定性度量
        with jt.no_grad():
            max_probs = jt.nn.softmax(input, dim=1).max(dim=1)
            confidence_uncertainty = 1.0 - max_probs
        
        # 可以选择不同的不确定性度量方式
        # 这里返回损失值作为默认的不确定性度量
        return entropy


class OHEMSampler:
    """
    Online Hard Example Mining (OHEM) 采样器
    用于在训练过程中动态选择困难样本
    支持基于损失值或不确定性的筛选策略
    """
    def __init__(self, hard_ratio=0.7, min_kept=256, selection_strategy='loss'):
        """
        Args:
            hard_ratio: 困难样本的比例，例如0.7表示选择70%的困难样本
            min_kept: 每个batch至少保留的样本数量
            selection_strategy: 选择策略，'loss' 或 'uncertainty' 或 'combined'
        """
        self.hard_ratio = hard_ratio
        self.min_kept = min_kept
        self.selection_strategy = selection_strategy
    
    def select_hard_examples(self, per_sample_losses, batch_size=None, uncertainty_scores=None):
        """
        根据损失值或不确定性选择困难样本
        
        Args:
            per_sample_losses: 每个样本的损失值，形状为 [batch_size]
            batch_size: 如果提供，会覆盖从per_sample_losses推断的batch_size
            uncertainty_scores: 每个样本的不确定性分数，形状为 [batch_size]（可选）
        
        Returns:
            selected_indices: 被选中的样本索引
            selected_losses: 被选中样本的损失值
            selection_scores: 用于选择的分数（损失值或不确定性分数）
        """
        if batch_size is None:
            batch_size = per_sample_losses.shape[0]
        
        # 计算要保留的样本数量
        num_kept = max(int(batch_size * self.hard_ratio), self.min_kept)
        num_kept = min(num_kept, batch_size)  # 不能超过总样本数
        
        # 根据选择策略确定排序依据
        if self.selection_strategy == 'loss':
            # 基于损失值选择
            selection_scores = per_sample_losses
        elif self.selection_strategy == 'uncertainty':
            # 基于不确定性选择
            if uncertainty_scores is None:
                raise ValueError("uncertainty_scores is required for 'uncertainty' selection strategy")
            selection_scores = uncertainty_scores
        elif self.selection_strategy == 'combined':
            # 结合损失值和不确定性
            if uncertainty_scores is None:
                raise ValueError("uncertainty_scores is required for 'combined' selection strategy")
            # 将损失值和不确定性归一化后组合
            normalized_losses = (per_sample_losses - per_sample_losses.min()) / (per_sample_losses.max() - per_sample_losses.min() + 1e-8)
            normalized_uncertainty = (uncertainty_scores - uncertainty_scores.min()) / (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)
            selection_scores = 0.5 * normalized_losses + 0.5 * normalized_uncertainty
        else:
            raise ValueError(f"Unknown selection_strategy: {self.selection_strategy}")
        
        # 选择分数最高的样本
        _, sorted_indices = jt.argsort(selection_scores, descending=True)
        selected_indices = sorted_indices[:num_kept]
        
        # 获取被选中样本的损失值和选择分数
        selected_losses = per_sample_losses[selected_indices]
        selected_scores = selection_scores[selected_indices]
        
        return selected_indices, selected_losses, selected_scores
    
    def apply_ohem(self, criterion, input, target, epoch, use_uncertainty=False, **kwargs):
        """
        应用OHEM到损失计算
        
        Args:
            criterion: 损失函数（应该支持get_per_sample_loss方法）
            input: 模型输出
            target: 真实标签
            epoch: 当前训练轮次
            use_uncertainty: 是否使用不确定性进行样本选择
            **kwargs: 传递给损失函数的其他参数
        
        Returns:
            loss: OHEM后的平均损失
            selected_indices: 被选中的样本索引（用于调试/分析）
            selection_info: 选择过程的详细信息
        """
        # 获取每个样本的独立损失
        per_sample_losses = criterion.get_per_sample_loss(input, target, epoch)
        
        # 获取不确定性分数（如果需要）
        uncertainty_scores = None
        if use_uncertainty or self.selection_strategy in ['uncertainty', 'combined']:
            if hasattr(criterion, 'get_uncertainty_scores'):
                uncertainty_scores = criterion.get_uncertainty_scores(input, target, epoch)
            else:
                # 如果损失函数没有不确定性方法，使用简单的熵计算
                with jt.no_grad():
                    probs = jt.nn.softmax(input, dim=1)
                    uncertainty_scores = -(probs * jt.log(probs + 1e-8)).sum(dim=1)
        
        # 选择困难样本
        selected_indices, selected_losses, selection_scores = self.select_hard_examples(
            per_sample_losses, uncertainty_scores=uncertainty_scores
        )
        
        # 计算选中样本的平均损失
        ohem_loss = selected_losses.mean()
        
        # 准备选择信息
        selection_info = {
            'strategy': self.selection_strategy,
            'total_samples': per_sample_losses.shape[0],
            'selected_samples': len(selected_indices),
            'avg_loss': per_sample_losses.mean().item(),
            'avg_selected_loss': selected_losses.mean().item(),
        }
        
        if uncertainty_scores is not None:
            selection_info.update({
                'avg_uncertainty': uncertainty_scores.mean().item(),
                'avg_selected_uncertainty': uncertainty_scores[selected_indices].mean().item(),
            })
        
        return ohem_loss, selected_indices, selection_info


# 使用示例函数
def example_usage():
    """
    展示如何在训练循环中使用OHEM和RWLDAMDRWLoss
    """
    # 假设的类别分布
    cls_num_list = [1000, 100, 50, 10, 5]  # 严重不平衡的数据集
    
    # 初始化损失函数
    criterion = RWLDAMDRWLoss(
        cls_num_list=cls_num_list,
        max_m=0.5,
        s=30,
        reweight_epoch=80,
        total_epochs=200
    )
    
    # 初始化不同策略的OHEM采样器
    ohem_loss_sampler = OHEMSampler(hard_ratio=0.7, min_kept=16, selection_strategy='loss')
    ohem_uncertainty_sampler = OHEMSampler(hard_ratio=0.7, min_kept=16, selection_strategy='uncertainty')
    ohem_combined_sampler = OHEMSampler(hard_ratio=0.7, min_kept=16, selection_strategy='combined')
    
    # 模拟训练循环
    batch_size = 32
    num_classes = len(cls_num_list)
    epoch = 50
    
    # 模拟数据
    input = jt.randn(batch_size, num_classes)
    target = jt.randint(0, num_classes, (batch_size,))
    
    print("=== 标准损失计算 ===")
    # 标准损失计算
    standard_loss = criterion(input, target, epoch)
    print(f"标准损失: {standard_loss.item():.4f}")
    
    print("\n=== 基于损失的OHEM ===")
    # 基于损失的OHEM
    ohem_loss, selected_indices, info = ohem_loss_sampler.apply_ohem(criterion, input, target, epoch)
    print(f"OHEM损失: {ohem_loss.item():.4f}")
    print(f"选中的样本数量: {len(selected_indices)}")
    print(f"选择策略: {info['strategy']}")
    print(f"平均损失: {info['avg_loss']:.4f} -> 选中样本平均损失: {info['avg_selected_loss']:.4f}")
    
    print("\n=== 基于不确定性的OHEM ===")
    # 基于不确定性的OHEM
    ohem_uncertainty_loss, uncertainty_indices, uncertainty_info = ohem_uncertainty_sampler.apply_ohem(criterion, input, target, epoch)
    print(f"不确定性OHEM损失: {ohem_uncertainty_loss.item():.4f}")
    print(f"选中的样本数量: {len(uncertainty_indices)}")
    print(f"平均不确定性: {uncertainty_info['avg_uncertainty']:.4f} -> 选中样本平均不确定性: {uncertainty_info['avg_selected_uncertainty']:.4f}")
    
    print("\n=== 组合策略OHEM ===")
    # 组合策略OHEM
    ohem_combined_loss, combined_indices, combined_info = ohem_combined_sampler.apply_ohem(criterion, input, target, epoch)
    print(f"组合OHEM损失: {ohem_combined_loss.item():.4f}")
    print(f"选中的样本数量: {len(combined_indices)}")
    print(f"策略: 损失 + 不确定性组合")
    
    print("\n=== 样本不确定性分析 ===")
    # 不确定性分析
    uncertainty_scores = criterion.get_uncertainty_scores(input, target, epoch)
    print(f"平均不确定性分数: {uncertainty_scores.mean().item():.4f}")
    print(f"最高不确定性分数: {uncertainty_scores.max().item():.4f}")
    print(f"最低不确定性分数: {uncertainty_scores.min().item():.4f}")


if __name__ == "__main__":
    example_usage()

