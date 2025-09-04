import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Dict
import logging
from collections import Counter, defaultdict
import numpy as np
import jittor as jt
from utils.dataset import ImageFolder2
from PIL import Image


class ImbalancedDatasetSampler:
    """类别不平衡采样器"""
    def __init__(self, samples: List[Tuple], minority_boost_ratio: float = 0.33, 
                 oversample_factor: float = 2.0, num_classes: int = 6):
        self.samples = samples
        self.minority_boost_ratio = minority_boost_ratio  # 少数类占比目标
        self.oversample_factor = oversample_factor  # 过采样倍数
        self.num_classes = num_classes
        
        # 统计各类别样本数量
        self.class_counts = self._count_classes()
        self.total_samples = len(samples)
        
        # 识别少数类和多数类
        self.minority_classes, self.majority_classes = self._identify_minority_majority()
        
        # 计算采样权重
        self.sample_weights = self._calculate_sample_weights()
        
        logging.info(f"原始类别分布: {dict(self.class_counts)}")
        logging.info(f"少数类: {self.minority_classes}")
        logging.info(f"多数类: {self.majority_classes}")
    
    def _count_classes(self) -> Counter:
        """统计各类别样本数量"""
        class_counts = Counter()
        for _, label in self.samples:
            class_counts[label] += 1
        return class_counts
    
    def _identify_minority_majority(self) -> Tuple[List[int], List[int]]:
        """识别少数类和多数类"""
        # 按样本数量排序
        sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1])
        
        # 取样本数最少的3个类作为少数类
        minority_classes = [cls for cls, _ in sorted_classes[:3]]
        majority_classes = [cls for cls, _ in sorted_classes[3:]]
        
        return minority_classes, majority_classes
    
    def _calculate_sample_weights(self) -> np.ndarray:
        """计算每个样本的采样权重"""
        weights = np.zeros(len(self.samples))
        
        # 计算少数类和多数类的总样本数
        minority_total = sum(self.class_counts[cls] for cls in self.minority_classes)
        majority_total = sum(self.class_counts[cls] for cls in self.majority_classes)
        
        # 计算目标采样比例：希望少数类占总采样的minority_boost_ratio
        # 如果少数类目标占比是0.33，那么多数类占比是0.67
        minority_target_ratio = self.minority_boost_ratio
        majority_target_ratio = 1.0 - minority_target_ratio
        
        # 当前少数类和多数类的实际占比
        minority_current_ratio = minority_total / self.total_samples
        majority_current_ratio = majority_total / self.total_samples
        
        # 计算需要的boost factor来达到目标比例
        minority_boost_factor = minority_target_ratio / minority_current_ratio if minority_current_ratio > 0 else 1.0
        majority_boost_factor = majority_target_ratio / majority_current_ratio if majority_current_ratio > 0 else 1.0
        
        logging.info(f"当前分布 - 少数类: {minority_current_ratio:.3f}, 多数类: {majority_current_ratio:.3f}")
        print(f"目标分布 - 少数类: {minority_target_ratio:.3f}, 多数类: {majority_target_ratio:.3f}")
        logging.info(f"目标分布 - 少数类: {minority_target_ratio:.3f}, 多数类: {majority_target_ratio:.3f}")
        print(f"boost因子 - 少数类: {minority_boost_factor:.3f}, 多数类: {majority_boost_factor:.3f}")
        logging.info(f"boost因子 - 少数类: {minority_boost_factor:.3f}, 多数类: {majority_boost_factor:.3f}")
        print(f"当前分布 - 少数类: {minority_current_ratio:.3f}, 多数类: {majority_current_ratio:.3f}")
        
        for i, (_, label) in enumerate(self.samples):
            if label in self.minority_classes:
                # 少数类：应用boost factor，并考虑类内不平衡
                class_weight = minority_boost_factor / self.class_counts[label]
                weights[i] = class_weight * self.oversample_factor
            else:
                # 多数类：应用较小的boost factor
                class_weight = majority_boost_factor / self.class_counts[label]
                weights[i] = class_weight
        
        return weights
    
    def get_balanced_batch_indices(self, batch_size: int) -> List[int]:
        """获取类别平衡的batch索引"""
        # 计算少数类在batch中的目标数量
        minority_batch_size = round(batch_size * self.minority_boost_ratio)
        majority_batch_size = batch_size - minority_batch_size
        
        indices = []
        
        # === 步骤1: 从少数类中平均采样 ===
        if len(self.minority_classes) > 0 and minority_batch_size > 0:
            # 计算每个少数类应分配的样本数
            samples_per_minority = minority_batch_size // len(self.minority_classes)
            extra_samples = minority_batch_size % len(self.minority_classes)
            
            # 为每个少数类采样
            for i, cls in enumerate(self.minority_classes):
                cls_indices = [idx for idx, (_, label) in enumerate(self.samples) if label == cls]
                if len(cls_indices) > 0:
                    # 前extra_samples个类别多分配1个样本
                    cls_sample_count = samples_per_minority + (1 if i < extra_samples else 0)
                    if len(cls_indices) >= cls_sample_count:
                        # 优先不重复采样
                        sampled_indices = np.random.choice(
                            cls_indices, 
                            size=cls_sample_count, 
                            replace=False
                        )
                    else:
                        # 先全取一遍，再补齐
                        sampled_indices = list(cls_indices)
                        needed = cls_sample_count - len(cls_indices)
                        if needed > 0:
                            sampled_indices += list(np.random.choice(cls_indices, size=needed, replace=True))
                    indices.extend(sampled_indices)
        
        # === 步骤2: 从其他类别中随机采样填充剩余位置 ===
        if majority_batch_size > 0:
            # 获取非少数类的所有样本索引
            majority_indices = [idx for idx, (_, label) in enumerate(self.samples) 
                              if label not in self.minority_classes]
            
            if len(majority_indices) >= majority_batch_size:
                # 从多数类中随机采样（不重复）
                sampled_majority = np.random.choice(
                    majority_indices, 
                    size=majority_batch_size, 
                    replace=False
                )
                indices.extend(sampled_majority.tolist())
            else:
                # 如果多数类样本不足，允许重复采样
                sampled_majority = np.random.choice(
                    majority_indices, 
                    size=majority_batch_size, 
                    replace=True
                )
                indices.extend(sampled_majority.tolist())
        
        # === 步骤3: 打乱batch内样本顺序 ===
        np.random.shuffle(indices)
        
        # === 步骤4: 验证结果 ===
        assert len(indices) == batch_size, f"Batch大小错误: 期望{batch_size}, 实际{len(indices)}"
        
        # 检查是否有重复样本（可选，调试用）
        # if len(indices) != len(set(indices)):
        #     print("Warning: 当前batch存在重复样本！")
        
        # 统计分布（用于调试）
        batch_labels = [self.samples[idx][1] for idx in indices]
        minority_count = sum(1 for label in batch_labels if label in self.minority_classes)
        majority_count = len(batch_labels) - minority_count
        
        actual_minority_ratio = minority_count / batch_size
        
        # 只在前几个batch输出详细信息，避免日志过多
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        self._batch_count += 1
        
        if self._batch_count <= 3:  # 只输出前3个batch的详细信息
            from collections import Counter
            label_counter = Counter(batch_labels)
            logging.info(f"Batch {self._batch_count}: 少数类{minority_count}/{batch_size}({actual_minority_ratio:.1%}), "
                        f"分布={dict(sorted(label_counter.items()))}")
        
        return indices
    
    
class BalancedImageFolder2(ImageFolder2):
    """支持类别平衡采样的数据加载器"""
    def __init__(self, root, samples, transform=None, batch_size=16, num_workers=8, 
                 shuffle=True, use_balanced_sampling=True, minority_boost_ratio=0.4, **kwargs):
        # 正确传递所有参数给父类
        super().__init__(root, samples, transform, batch_size=batch_size, 
                        num_workers=num_workers, shuffle=shuffle, **kwargs)
        
        self.use_balanced_sampling = use_balanced_sampling
        self.target_batch_size = batch_size  # 明确保存目标batch_size
        self.minority_boost_ratio = minority_boost_ratio
        
        if use_balanced_sampling:
            self.sampler = ImbalancedDatasetSampler(
                samples, minority_boost_ratio=minority_boost_ratio
            )
            # 打印采样器的详细信息用于调试
            logging.info(f"=== 平衡采样配置 ===")
            logging.info(f"目标batch_size: {self.target_batch_size}")
            logging.info(f"少数类目标占比: {minority_boost_ratio}")
            print(f"=== 平衡采样配置 ===")
            print(f"目标batch_size: {self.target_batch_size}")
            print(f"少数类目标占比: {minority_boost_ratio}")
    
    def __iter__(self):
        if self.use_balanced_sampling:
            # 使用平衡采样
            total_batches = len(self.samples) // self.target_batch_size
            for batch_idx in range(total_batches):
                batch_indices = self.sampler.get_balanced_batch_indices(self.target_batch_size)
                batch_data = []
                
                for idx in batch_indices:
                    path, label = self.samples[idx]
                    # 根据ImageFolder2的实现方式加载图片
                    image_path = os.path.join(self.root, path)
                    from PIL import Image
                    image = Image.open(image_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    batch_data.append((image, label))
                
                # 验证batch大小
                assert len(batch_data) == self.target_batch_size, \
                    f"Batch size mismatch: expected {self.target_batch_size}, got {len(batch_data)}"
                
                # 只在前3个batch打印详细分布信息
                # if batch_idx < 3:
                #     batch_labels = [item[1] for item in batch_data]
                #     self._log_batch_distribution(batch_idx + 1, batch_labels)
                
                # 组织batch - 处理jittor tensor转换
                try:
                    images = jt.stack([jt.array(np.array(item[0])) for item in batch_data])
                    labels = jt.array([item[1] for item in batch_data])
                except:
                    # 如果直接转换失败，尝试其他方式
                    images = jt.stack([item[0] if hasattr(item[0], 'shape') else jt.array(np.array(item[0])) for item in batch_data])
                    labels = jt.array([item[1] for item in batch_data])
                
                yield images, labels
        else:
            # 使用原始采样方式
            yield from super().__iter__()
    
    def _log_batch_distribution(self, batch_num: int, labels: List[int]):
        """记录batch的类别分布"""
        from collections import Counter
        label_counts = Counter(labels)
        total = len(labels)
        
        minority_count = sum(label_counts[cls] for cls in self.sampler.minority_classes)
        majority_count = sum(label_counts[cls] for cls in self.sampler.majority_classes)
        
        print(f"Batch {batch_num} 详细分布: {dict(sorted(label_counts.items()))}")
        print(f"Batch {batch_num} 占比: 少数类={minority_count}/{total}({minority_count/total:.1%}), "
              f"多数类={majority_count}/{total}({majority_count/total:.1%})")
        
        logging.info(f"Batch {batch_num} 分布: 总数={total}, 少数类={minority_count}({minority_count/total:.2%}), "
                    f"多数类={majority_count}({majority_count/total:.2%})")
        logging.info(f"Batch {batch_num} 详细: {dict(label_counts)}")
            
            
            
if __name__ == "__main__":
    from dataset import load_train_val_samples
    import os
    from jittor.transform import (
    Compose, Resize, CenterCrop, RandomCrop, RandomRotation, RandomVerticalFlip, 
    RandomHorizontalFlip, ToTensor, ImageNormalize, RandomResizedCrop, RandomAffine, 
    ColorJitter
)
    root_path = "/root/workspace/Jittor/DATASET/TrainSet"
    # 加载真实训练样本
    train_samples, _ = load_train_val_samples(root_path)
    print(f"训练样本数: {len(train_samples)}")
    
    transform_train = Compose([
        RandomResizedCrop((512, 512), scale=(0.8, 1.0), ratio=(3/4, 4/3)), 
        RandomCrop((384, 384)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        Resize((224, 224)),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = Compose([
        Resize((512, 512)),
        CenterCrop((384, 384)),
        Resize((224, 224)),
        ToTensor(),
        ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    batch_size = 96
    num_batches = 6

    print("\n===== 原始ImageFolder2采样 =====")
    loader = ImageFolder2(
        root=os.path.join(root_path, 'images/train'), samples=train_samples, transform=transform_train, batch_size=batch_size, num_workers=0, shuffle=True
    )
    for i, (images, labels) in zip(range(num_batches), loader):
        print(f"Batch {i+1}: ", labels.numpy().tolist())

    print("\n===== BalancedImageFolder2类别平衡采样 =====")
    from utils.sampler import BalancedImageFolder2
    balanced_loader = BalancedImageFolder2(
        root=os.path.join(root_path, 'images/train'), samples=train_samples, transform=transform_train, batch_size=batch_size, num_workers=0, shuffle=True, use_balanced_sampling=True, minority_boost_ratio=0.40
    )
    for i, (images, labels) in zip(range(num_batches), balanced_loader):
        print(f"Batch {i+1}: ", labels.numpy().tolist())
            
            
            