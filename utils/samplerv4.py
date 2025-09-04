import os
from PIL import Image
import jittor as jt
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize
from sklearn.model_selection import KFold
import logging


class ManualBalancedSampler:
    """手动指定各类别过采样系数的采样器"""
    
    def __init__(self, samples: List[Tuple], class_oversample_ratios: Dict[int, float]):
        """
        Args:
            samples: 样本列表 [(path, label), ...]
            class_oversample_ratios: 各类别的过采样系数 {class_id: ratio}
                                   ratio=1.0表示不过采样，ratio=2.0表示翻倍等
        """
        self.samples = samples
        self.class_oversample_ratios = class_oversample_ratios
        
        # 预计算类别信息
        self.class_counts = Counter(label for _, label in samples)
        self.num_classes = len(self.class_counts)
        self.total_samples = len(samples)
        
        # 按类别分组样本索引
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(samples):
            self.class_to_indices[label].append(idx)
        
        # 计算各类别的目标样本数
        self._calculate_target_counts()
        
    def _calculate_target_counts(self):
        """计算各类别的目标样本数"""
        self.target_class_counts = {}
        self.total_target_samples = 0
        
        print(f"\n📝 手动过采样配置:")
        for cls in sorted(self.class_counts.keys()):
            original_count = self.class_counts[cls]
            oversample_ratio = self.class_oversample_ratios.get(cls, 1.0)
            target_count = int(original_count * oversample_ratio)
            
            self.target_class_counts[cls] = target_count
            self.total_target_samples += target_count
            
            print(f"  类别 {cls}: {original_count:3d} → {target_count:3d} (系数: {oversample_ratio:.2f}x)")
        
        print(f"  总计: {self.total_samples} → {self.total_target_samples} (+{self.total_target_samples - self.total_samples})")
        
    def generate_epoch_indices(self) -> List[int]:
        """生成一个epoch的所有索引"""
        indices = []
        
        for cls in sorted(self.class_counts.keys()):
            cls_indices = self.class_to_indices[cls]
            target_count = self.target_class_counts[cls]
            original_count = len(cls_indices)
            
            if target_count <= original_count:
                # 不需要过采样，随机选择目标数量
                selected_indices = np.random.choice(cls_indices, size=target_count, replace=False)
            else:
                # 需要过采样
                # 首先包含所有原始样本
                selected_indices = list(cls_indices)
                # 然后随机重复采样到目标数量
                additional_needed = target_count - original_count
                additional_indices = np.random.choice(cls_indices, size=additional_needed, replace=True)
                selected_indices.extend(additional_indices.tolist())
            
            indices.extend(selected_indices)
        
        # 打乱整个epoch的顺序
        np.random.shuffle(indices)
        return indices


class EfficientBalancedSampler:
    """高效的类别平衡采样器 - 确保全数据覆盖且过采样最少"""
    
    def __init__(self, samples: List[Tuple], target_minority_ratio: float = 0.3):
        self.samples = samples
        self.target_minority_ratio = target_minority_ratio
        
        # 预计算类别信息
        self.class_counts = Counter(label for _, label in samples)
        self.num_classes = len(self.class_counts)
        self.total_samples = len(samples)
        
        # 按类别分组样本索引 - 只计算一次
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(samples):
            self.class_to_indices[label].append(idx)
        
        # 识别少数类和多数类
        sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1])
        mid_point = self.num_classes // 2
        self.minority_classes = [cls for cls, _ in sorted_classes[:mid_point]]
        self.majority_classes = [cls for cls, _ in sorted_classes[mid_point:]]
        
        # 计算最优的epoch增强策略
        self._calculate_optimal_strategy()
        
    def _calculate_optimal_strategy(self):
        """计算最优策略：最小过采样 + 全数据覆盖"""
        minority_total = sum(self.class_counts[cls] for cls in self.minority_classes)
        majority_total = sum(self.class_counts[cls] for cls in self.majority_classes)
        
        # 策略：通过少量过采样达到目标比例，而不是大量复制
        current_minority_ratio = minority_total / self.total_samples
        
        if current_minority_ratio < self.target_minority_ratio:
            # 计算需要额外采样的少数类样本数
            # 目标：minority_total + extra = target_ratio * (total + extra)
            # 解方程得到 extra
            extra_needed = (self.target_minority_ratio * self.total_samples - minority_total) / (1 - self.target_minority_ratio)
            self.extra_minority_samples = max(0, int(extra_needed))
        else:
            self.extra_minority_samples = 0
            
        self.total_epoch_samples = self.total_samples + self.extra_minority_samples
        
        logging.info(f"平衡策略: 原始{self.total_samples}, 少数类额外{self.extra_minority_samples}, "
                    f"总计{self.total_epoch_samples}样本")
    
    def generate_epoch_indices(self) -> List[int]:
        """生成一个epoch的所有索引 - 保证全覆盖 + 最少过采样"""
        indices = list(range(self.total_samples))  # 所有原始样本
        
        # 添加少数类的额外采样
        if self.extra_minority_samples > 0:
            minority_indices = []
            for cls in self.minority_classes:
                minority_indices.extend(self.class_to_indices[cls])
            
            # 随机选择需要额外采样的少数类样本
            extra_indices = np.random.choice(
                minority_indices, 
                size=self.extra_minority_samples, 
                replace=True
            )
            indices.extend(extra_indices.tolist())
        
        # 打乱整个epoch的顺序
        np.random.shuffle(indices)
        return indices


class BalanceImageFolder2(Dataset):
    def __init__(self, root, samples=None, transform=None, use_balanced_sampling=False, 
                 target_minority_ratio=0.3, class_oversample_ratios=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        self.samples = samples
        self.total_len = len(self.samples)
        self.use_balanced_sampling = use_balanced_sampling
        self.class_oversample_ratios = class_oversample_ratios
        
        # 初始化采样器
        if use_balanced_sampling and samples:
            if class_oversample_ratios is not None:
                # 使用手动指定的过采样系数
                self.balanced_sampler = ManualBalancedSampler(samples, class_oversample_ratios)
                self.total_len = self.balanced_sampler.total_target_samples
                
            else:
                # 使用自动平衡采样器
                self.balanced_sampler = EfficientBalancedSampler(samples, target_minority_ratio)
                self.total_len = self.balanced_sampler.total_epoch_samples
            
        else:
            # 不使用平衡采样时的默认长度
            self.total_len = len(self.samples)
        
        # 设置Jittor框架必需的属性
        self.real_len = self.total_len
        self.real_batch_size = self.batch_size  # 添加real_batch_size属性
        self.batch_len = (self.total_len + self.batch_size - 1) // self.batch_size  # 计算batch数量
        
        # 设置数据集属性
        self.set_attrs(total_len=self.total_len)

    def __getitem__(self, idx):
        # 如果使用平衡采样，需要将虚拟索引映射到真实索引
        if self.use_balanced_sampling and hasattr(self, 'epoch_indices'):
            actual_idx = self.epoch_indices[idx % len(self.epoch_indices)]
        else:
            actual_idx = idx
            
        image_path, label = os.path.join(self.root, self.samples[actual_idx][0]), self.samples[actual_idx][1]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.samples[actual_idx][0] if label is None else label
        return jt.array(image), label
    
    def __len__(self):
        return self.total_len
    
    def _get_index_list(self):
        """重写索引生成逻辑"""
        if self.use_balanced_sampling:
            # 生成平衡的epoch索引
            self.epoch_indices = self.balanced_sampler.generate_epoch_indices()
            # 返回虚拟索引列表
            return list(range(len(self.epoch_indices)))
        else:
            # 使用父类的默认实现
            return super()._get_index_list()
