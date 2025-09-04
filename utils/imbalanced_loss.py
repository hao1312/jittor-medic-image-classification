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
        
    def execute(self, input, target):
        """
        Args:
            input: 模型输出的logits，形状为 [batch_size, num_classes]
            target: 真实标签，形状为 [batch_size]
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
        return nn.CrossEntropyLoss(weight=self.weight)(self.s * output, target)


class FocalLoss(nn.Module):
    """
    Focal Loss for Jittor
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def execute(self, input, target):
        # 先计算标准的交叉熵损失
        if self.weight is not None:
            ce_loss_fn = nn.CrossEntropyLoss(weight=self.weight)
        else:
            ce_loss_fn = nn.CrossEntropyLoss()
        
        # 计算每个样本的损失
        log_probs = nn.log_softmax(input, dim=1)
        target_one_hot = jt.zeros(input.shape)
        for i in range(input.shape[0]):
            target_one_hot[i, target[i]] = 1.0
        
        # 计算负对数似然
        nll = -(log_probs * target_one_hot).sum(dim=1)
        
        # 应用权重
        if self.weight is not None:
            weights_expanded = self.weight[target]
            nll = nll * weights_expanded
        
        # 计算focal项
        probs = jt.exp(log_probs)
        pt = (probs * target_one_hot).sum(dim=1)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * nll
        return focal_loss.mean()


class CBLoss(nn.Module):
    """
    Class-Balanced Loss for Jittor
    """
    def __init__(self, cls_num_list, beta=0.9999, gamma=2.0, loss_type='focal'):
        super(CBLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # 计算有效样本数
        effective_num = 1.0 - np.power(beta, cls_num_list)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(cls_num_list)
        self.weights = jt.array(weights).float()
        
    def execute(self, input, target):
        if self.loss_type == 'focal':
            cb_loss = FocalLoss(weight=self.weights, gamma=self.gamma)(input, target)
        elif self.loss_type == 'ce':
            cb_loss = nn.CrossEntropyLoss(weight=self.weights)(input, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        return cb_loss


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
    
    def execute(self, input, target, epoch):
        """
        根据当前epoch决定使用哪种损失
        
        Args:
            input: 模型输出
            target: 真实标签  
            epoch: 当前训练轮次
        """
        if epoch < self.reweight_epoch:
            # 第一阶段：使用普通LDAM损失
            return self.ldam_loss(input, target)
        else:
            # 第二阶段：使用重加权LDAM损失
            return self.ldam_loss_reweight(input, target)


class ReweightedCrossEntropyLoss(nn.Module):
    """
    带重加权的交叉熵损失
    """
    def __init__(self, cls_num_list, reweight_type='inverse'):
        super(ReweightedCrossEntropyLoss, self).__init__()
        self.weights = get_class_weights(cls_num_list, reweight_type)
        
    def execute(self, input, target):
        return nn.CrossEntropyLoss(weight=self.weights)(input, target)


class CEDRWLoss(nn.Module):
    """
    CE-DRW (Cross Entropy with Delayed Re-Weighting) 损失
    """
    def __init__(self, cls_num_list, reweight_epoch=160, total_epochs=200, 
                 reweight_type='inverse'):
        super(CEDRWLoss, self).__init__()
        
        self.cls_num_list = cls_num_list
        self.reweight_epoch = reweight_epoch
        self.total_epochs = total_epochs
        
        # 普通交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 重加权交叉熵损失
        self.reweight_weights = get_class_weights(cls_num_list, reweight_type)
        self.ce_loss_reweight = nn.CrossEntropyLoss(weight=self.reweight_weights)
    
    def execute(self, input, target, epoch):
        """
        根据当前epoch决定使用哪种损失
        """
        if epoch < self.reweight_epoch:
            # 第一阶段：使用普通CE损失
            return self.ce_loss(input, target)
        else:
            # 第二阶段：使用重加权CE损失
            return self.ce_loss_reweight(input, target)


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
    
    def execute(self, input, target, epoch):
        """
        根据当前epoch决定使用哪种损失
        """
        if epoch < self.reweight_epoch:
            # 第一阶段：使用普通LDAM损失
            return self.ldam_loss(input, target)
        else:
            # 第二阶段：使用重加权LDAM损失
            return self.ldam_loss_reweight(input, target)


class CELDAMDRWLoss(nn.Module):
    """
    CE LDAM-DRW (Cross Entropy LDAM-DRW) 损失
    将LDAM的边距机制应用到交叉熵损失中，结合延迟重加权
    """
    def __init__(self, cls_num_list, max_m=0.5, reweight_epoch=160, 
                 total_epochs=200, reweight_type='inverse'):
        super(CELDAMDRWLoss, self).__init__()
        
        self.cls_num_list = cls_num_list
        self.num_classes = len(cls_num_list)
        self.reweight_epoch = reweight_epoch
        self.total_epochs = total_epochs
        
        # 计算每个类别的边距
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = jt.array(m_list).float()
        
        # 普通交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 重加权交叉熵损失
        self.reweight_weights = get_class_weights(cls_num_list, reweight_type)
        self.ce_loss_reweight = nn.CrossEntropyLoss(weight=self.reweight_weights)
    
    def execute(self, input, target, epoch):
        """
        应用LDAM边距后计算交叉熵损失
        """
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        
        # 创建one-hot编码
        index = jt.zeros([batch_size, num_classes])
        target_long = target.long()
        for i in range(batch_size):
            index[i, target_long[i]] = 1.0
        
        # 为每个样本应用对应类别的边距
        margins = self.m_list[target_long]  # [batch_size]
        margins = margins.unsqueeze(1).expand(batch_size, num_classes)  # [batch_size, num_classes]
        
        # 只对真实类别应用边距（降低其logit值）
        adjusted_input = input - margins * index
        
        # 根据当前epoch决定使用哪种交叉熵损失
        if epoch < self.reweight_epoch:
            # 第一阶段：使用普通CE损失
            return self.ce_loss(adjusted_input, target)
        else:
            # 第二阶段：使用重加权CE损失
            return self.ce_loss_reweight(adjusted_input, target)


def create_imbalanced_criterion(loss_type, cls_num_list, **kwargs):
    """
    创建不平衡数据损失函数的工厂函数
    
    Args:
        loss_type: 损失类型 ('ldam_drw', 'rw_ldam_drw', 'ce_ldam_drw', 'ce_drw', 'focal', 'cb_focal', 'cb_ce', 'rw_ce')
        cls_num_list: 每个类别的样本数量列表
        **kwargs: 其他参数
    """
    if loss_type == 'ldam_drw':
        # 为了向后兼容，保持原来的LDAMDRWLoss
        return LDAMDRWLoss(cls_num_list, **kwargs)
    elif loss_type == 'rw_ldam_drw':
        # 明确的重加权LDAM-DRW
        return RWLDAMDRWLoss(cls_num_list, **kwargs)
    elif loss_type == 'ce_ldam_drw':
        # CE LDAM-DRW变体
        return CELDAMDRWLoss(cls_num_list, **kwargs)
    elif loss_type == 'ce_drw':
        return CEDRWLoss(cls_num_list, **kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'cb_focal':
        return CBLoss(cls_num_list, loss_type='focal', **kwargs)
    elif loss_type == 'cb_ce':
        return CBLoss(cls_num_list, loss_type='ce', **kwargs)
    elif loss_type == 'rw_ce':
        return ReweightedCrossEntropyLoss(cls_num_list, **kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# 乳腺癌数据集的典型配置
class BreastCancerLossConfig:
    """
    乳腺癌超声图像分类的损失函数配置
    """
    
    @staticmethod
    def get_config(dataset_type='BUSI'):
        """
        获取不同数据集的配置
        
        Args:
            dataset_type: 数据集类型 ('BUSI', 'custom')
        """
        configs = {
            'BUSI': {
                # BUSI数据集通常有良性、恶性、正常三类
                'cls_num_list': [437, 210, 133],  # 示例分布：正常, 良性, 恶性
                'max_m': 0.5,
                's': 30,
                'reweight_epoch': 80,  # 在第80个epoch开始重加权
                'total_epochs': 200,
                'reweight_type': 'inverse',
                'focal_gamma': 2.0,
                'focal_alpha': 1.0,
                'cb_beta': 0.9999
            },
            'custom': {
                # 自定义配置，需要根据实际数据调整
                'cls_num_list': [1000, 500, 200],
                'max_m': 0.5,
                's': 30,
                'reweight_epoch': 100,
                'total_epochs': 250,
                'reweight_type': 'inverse',
                'focal_gamma': 2.0,
                'focal_alpha': 1.0,
                'cb_beta': 0.9999
            }
        }
        return configs.get(dataset_type, configs['custom'])


# 使用示例和对比实验
if __name__ == "__main__":
    # 模拟乳腺癌数据集的不平衡分布
    config = BreastCancerLossConfig.get_config('BUSI')
    cls_num_list = config['cls_num_list']
    num_classes = len(cls_num_list)
    
    print("=== 乳腺癌图像分类不平衡损失函数对比实验 ===")
    print(f"类别分布: 正常={cls_num_list[0]}, 良性={cls_num_list[1]}, 恶性={cls_num_list[2]}")
    print(f"不平衡比: {max(cls_num_list)/min(cls_num_list):.2f}")
    print()
    
    # 创建不同的损失函数
    losses = {
        'CE': nn.CrossEntropyLoss(),
        'RW-CE': create_imbalanced_criterion('rw_ce', cls_num_list),
        'CE-DRW': create_imbalanced_criterion('ce_drw', cls_num_list, 
                                            reweight_epoch=config['reweight_epoch'],
                                            total_epochs=config['total_epochs'],
                                            reweight_type=config['reweight_type']),
        'RW-LDAM-DRW': create_imbalanced_criterion('rw_ldam_drw', cls_num_list,
                                                  max_m=config['max_m'],
                                                  s=config['s'],
                                                  reweight_epoch=config['reweight_epoch'],
                                                  total_epochs=config['total_epochs'],
                                                  reweight_type=config['reweight_type']),
        'CE-LDAM-DRW': create_imbalanced_criterion('ce_ldam_drw', cls_num_list,
                                                  max_m=config['max_m'],
                                                  reweight_epoch=config['reweight_epoch'],
                                                  total_epochs=config['total_epochs'],
                                                  reweight_type=config['reweight_type']),
        'Focal': create_imbalanced_criterion('focal', cls_num_list, 
                                           gamma=config['focal_gamma'], 
                                           alpha=config['focal_alpha']),
        'CB-Focal': create_imbalanced_criterion('cb_focal', cls_num_list, 
                                              beta=config['cb_beta'], 
                                              gamma=config['focal_gamma'])
    }
    
    # 模拟数据
    batch_size = 32
    logits = jt.randn(batch_size, num_classes)
    # 模拟不平衡的标签分布（更多正常样本，较少恶性样本）
    labels = jt.concat([
        jt.zeros(16),  # 16个正常样本
        jt.ones(10),   # 10个良性样本  
        jt.ones(6) * 2 # 6个恶性样本
    ]).long()
    
    print("=== 第一阶段损失对比 (epoch=50) ===")
    epoch = 50
    for name, criterion in losses.items():
        if name in ['CE-DRW', 'RW-LDAM-DRW', 'CE-LDAM-DRW']:
            loss = criterion(logits, labels, epoch)
        else:
            loss = criterion(logits, labels)
        print(f"{name:12s}: {loss.item():.4f}")
    
    print("\n=== 第二阶段损失对比 (epoch=150) ===")
    epoch = 150
    for name, criterion in losses.items():
        if name in ['CE-DRW', 'RW-LDAM-DRW', 'CE-LDAM-DRW']:
            loss = criterion(logits, labels, epoch)
        else:
            loss = criterion(logits, labels)
        print(f"{name:12s}: {loss.item():.4f}")
    
    # 显示RW-LDAM-DRW的边距和权重信息
    rw_ldam_criterion = losses['RW-LDAM-DRW']
    print(f"\n=== RW-LDAM-DRW 边距值 ===")
    for i, (num, margin) in enumerate(zip(cls_num_list, rw_ldam_criterion.ldam_loss.m_list)):
        class_names = ['正常', '良性', '恶性']
        print(f"{class_names[i]}: 样本数={num}, 边距={margin:.4f}")
    
    print(f"\n=== RW-LDAM-DRW 重加权权重 ===")
    for i, weight in enumerate(rw_ldam_criterion.reweight_weights):
        class_names = ['正常', '良性', '恶性']
        print(f"{class_names[i]}: 权重={weight:.4f}")
    
    # 显示CE-LDAM-DRW的边距和权重信息
    ce_ldam_criterion = losses['CE-LDAM-DRW']
    print(f"\n=== CE-LDAM-DRW 边距值 ===")
    for i, (num, margin) in enumerate(zip(cls_num_list, ce_ldam_criterion.m_list)):
        class_names = ['正常', '良性', '恶性']
        print(f"{class_names[i]}: 样本数={num}, 边距={margin:.4f}")
    
    print(f"\n=== CE-LDAM-DRW 重加权权重 ===")
    for i, weight in enumerate(ce_ldam_criterion.reweight_weights):
        class_names = ['正常', '良性', '恶性']
        print(f"{class_names[i]}: 权重={weight:.4f}")
    
    print(f"\n=== RW-LDAM-DRW vs CE-LDAM-DRW 对比分析 ===")
    print("RW-LDAM-DRW (标准重加权LDAM-DRW):")
    print("  - 第一阶段: 使用普通LDAM损失 (带缩放因子s)")
    print("  - 第二阶段: 使用重加权LDAM损失 (LDAM + 类别权重)")
    print("  - 特点: 保持LDAM的所有特性，包括缩放因子")
    print()
    print("CE-LDAM-DRW (交叉熵LDAM-DRW):")
    print("  - 第一阶段: 使用带LDAM边距的交叉熵损失")
    print("  - 第二阶段: 使用带LDAM边距的重加权交叉熵损失")
    print("  - 特点: 简化版本，只使用边距机制，不使用缩放因子")
    print()
    print("=== 实验建议 ===")
    print("1. RW-LDAM-DRW: 适用于极度不平衡数据，需要强化少数类学习")
    print("2. CE-LDAM-DRW: 计算更简单，适用于中等不平衡数据")
    print("3. 两者都可以通过调整 reweight_epoch 控制重加权时机")
    print("4. 建议在验证集上比较两种方法的性能差异")
