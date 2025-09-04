"""
自定义Jittor Transform实现
实现GridDistortion、ElasticTransform、CoarseDropout等高级数据增强
"""

import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Tuple, Optional, Union, List
import jittor as jt


class GridDistortion:
    """
    Grid distortion transformation similar to Albumentations GridDistortion
    
    Args:
        num_steps (int): number of grid cells on each side
        distort_limit (float): distortion factor in range [0, 1]
        interpolation: PIL interpolation method
        border_mode: border handling mode
        p (float): probability of applying the transform
    """
    def __init__(self, num_steps: int = 5, distort_limit: float = 0.3, 
                 interpolation=Image.BILINEAR, border_mode=cv2.BORDER_REFLECT_101, p: float = 0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply grid distortion to image"""
        if random.random() > self.p:
            return img
            
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be PIL Image")
            
        # Convert PIL to numpy array
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Create grid
        x_step = width // self.num_steps
        y_step = height // self.num_steps
        
        # Generate random distortion
        xx = np.arange(0, width, x_step)
        yy = np.arange(0, height, y_step)
        
        # Add boundary points
        xx = np.concatenate([xx, [width]])
        yy = np.concatenate([yy, [height]])
        
        # Create distorted grid
        xx_distorted = xx + np.random.uniform(-self.distort_limit * x_step, 
                                            self.distort_limit * x_step, len(xx))
        yy_distorted = yy + np.random.uniform(-self.distort_limit * y_step, 
                                            self.distort_limit * y_step, len(yy))
        
        # Ensure boundary points remain fixed
        xx_distorted[0] = 0
        xx_distorted[-1] = width
        yy_distorted[0] = 0
        yy_distorted[-1] = height
        
        # Create meshgrid
        map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Interpolate distortion
        for i in range(len(yy) - 1):
            for j in range(len(xx) - 1):
                # Define region
                y1, y2 = yy[i], yy[i + 1]
                x1, x2 = xx[j], xx[j + 1]
                
                # Define distorted corners
                corners_x = [xx_distorted[j], xx_distorted[j + 1], 
                           xx_distorted[j], xx_distorted[j + 1]]
                corners_y = [yy_distorted[i], yy_distorted[i], 
                           yy_distorted[i + 1], yy_distorted[i + 1]]
                
                # Apply bilinear interpolation in this region
                region_mask = (map_y >= y1) & (map_y < y2) & (map_x >= x1) & (map_x < x2)
                if not np.any(region_mask):
                    continue
                    
                # Calculate interpolation weights
                y_region = map_y[region_mask]
                x_region = map_x[region_mask]
                
                # Normalize coordinates to [0, 1]
                u = (x_region - x1) / max(x2 - x1, 1)
                v = (y_region - y1) / max(y2 - y1, 1)
                
                # Bilinear interpolation
                new_x = (corners_x[0] * (1 - u) * (1 - v) + 
                        corners_x[1] * u * (1 - v) + 
                        corners_x[2] * (1 - u) * v + 
                        corners_x[3] * u * v)
                new_y = (corners_y[0] * (1 - u) * (1 - v) + 
                        corners_y[1] * u * (1 - v) + 
                        corners_y[2] * (1 - u) * v + 
                        corners_y[3] * u * v)
                
                map_x[region_mask] = new_x
                map_y[region_mask] = new_y
        
        # Apply remapping
        if len(img_array.shape) == 3:
            distorted = cv2.remap(img_array, map_x, map_y, 
                                cv2.INTER_LINEAR, borderMode=self.border_mode)
        else:
            distorted = cv2.remap(img_array, map_x, map_y, 
                                cv2.INTER_LINEAR, borderMode=self.border_mode)
        
        return Image.fromarray(distorted)


class ElasticTransform:
    """
    Elastic transformation similar to Albumentations ElasticTransform
    
    Args:
        alpha (float): scaling factor for the random displacement field
        sigma (float): standard deviation for Gaussian filter applied to displacement
        alpha_affine (float): scaling factor for affine transformation
        interpolation: PIL interpolation method
        border_mode: border handling mode
        p (float): probability of applying the transform
    """
    def __init__(self, alpha: float = 1, sigma: float = 50, alpha_affine: float = 50,
                 interpolation=Image.BILINEAR, border_mode=cv2.BORDER_REFLECT_101, p: float = 0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply elastic transformation to image"""
        if random.random() > self.p:
            return img
            
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be PIL Image")
            
        # Convert PIL to numpy array
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), 
                           self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), 
                           self.sigma, mode="constant", cval=0) * self.alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply elastic deformation
        x_new = (x + dx).astype(np.float32)
        y_new = (y + dy).astype(np.float32)
        
        # Apply remapping
        if len(img_array.shape) == 3:
            transformed = cv2.remap(img_array, x_new, y_new, 
                                  cv2.INTER_LINEAR, borderMode=self.border_mode)
        else:
            transformed = cv2.remap(img_array, x_new, y_new, 
                                  cv2.INTER_LINEAR, borderMode=self.border_mode)
        
        return Image.fromarray(transformed)


class CoarseDropout:
    """
    CoarseDropout transformation similar to Albumentations CoarseDropout
    Randomly removes rectangular regions from the image
    
    Args:
        max_holes (int): maximum number of regions to zero out
        max_height (int): maximum height of the hole
        max_width (int): maximum width of the hole
        min_holes (int): minimum number of regions to zero out
        min_height (int): minimum height of the hole
        min_width (int): minimum width of the hole
        fill_value (int or float): value to fill the holes with
        mask_fill_value (int or float): value to fill the holes in mask
        p (float): probability of applying the transform
    """
    def __init__(self, max_holes: int = 8, max_height: int = 8, max_width: int = 8,
                 min_holes: int = 1, min_height: int = 1, min_width: int = 1,
                 fill_value: Union[int, float] = 0, mask_fill_value: Union[int, float] = 0,
                 p: float = 0.5):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes
        self.min_height = min_height
        self.min_width = min_width
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply coarse dropout to image"""
        if random.random() > self.p:
            return img
            
        if not isinstance(img, Image.Image):
            raise TypeError("Input should be PIL Image")
            
        # Convert PIL to numpy array
        img_array = np.array(img).copy()
        height, width = img_array.shape[:2]
        
        # Determine number of holes
        num_holes = random.randint(self.min_holes, self.max_holes)
        
        for _ in range(num_holes):
            # Random hole dimensions
            hole_height = random.randint(self.min_height, min(self.max_height, height))
            hole_width = random.randint(self.min_width, min(self.max_width, width))
            
            # Random hole position
            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            
            # Fill the hole
            if len(img_array.shape) == 3:
                img_array[y1:y2, x1:x2, :] = self.fill_value
            else:
                img_array[y1:y2, x1:x2] = self.fill_value
        
        return Image.fromarray(img_array)


class RandomChoice:
    """
    Apply single transformation randomly picked from a list
    Similar to Albumentations OneOf
    
    Args:
        transforms (list): list of transformations
        p (float): probability of applying any transform
    """
    def __init__(self, transforms: List, p: float = 0.5):
        self.transforms = transforms
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply one random transformation"""
        if random.random() > self.p:
            return img
            
        if not self.transforms:
            return img
            
        # Choose random transform
        transform = random.choice(self.transforms)
        return transform(img)


class RandomApply:
    """
    Apply a list of transformations with given probability
    支持两种模式：
    1. 全部应用模式 (apply_all=True): 以概率p应用所有变换
    2. 概率应用模式 (apply_all=False): 每个变换都以概率p独立应用
    
    Args:
        transforms (list): list of transformations
        p (float): probability of applying the transforms
        apply_all (bool): if True, apply all transforms with probability p;
                         if False, apply each transform independently with probability p
    """
    def __init__(self, transforms: List, p: float = 0.5, apply_all: bool = True):
        self.transforms = transforms
        self.p = p
        self.apply_all = apply_all
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply transforms with probability p"""
        if self.apply_all:
            # 原始模式：以概率p应用所有变换
            if random.random() > self.p:
                return img
            for transform in self.transforms:
                img = transform(img)
        else:
            # 新模式：每个变换独立以概率p应用
            for transform in self.transforms:
                if random.random() <= self.p:
                    img = transform(img)
        return img


# Jittor兼容的包装器
class JittorTransformWrapper:
    """
    Wrapper to make custom transforms compatible with Jittor transform pipeline
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        """Apply transform and ensure compatibility with Jittor pipeline"""
        if isinstance(img, jt.Var):
            # Convert Jittor tensor to PIL Image
            img_np = img.numpy()
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:  # CHW format
                img_np = img_np.transpose(1, 2, 0)
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Apply transform
            transformed_pil = self.transform(img_pil)
            
            # Convert back to Jittor tensor
            transformed_np = np.array(transformed_pil)
            if transformed_np.ndim == 3:
                transformed_np = transformed_np.transpose(2, 0, 1)  # HWC to CHW
            transformed_np = transformed_np.astype(np.float32) / 255.0
            return jt.array(transformed_np)
        
        elif isinstance(img, np.ndarray):
            # Handle numpy array
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # CHW format
                img = img.transpose(1, 2, 0)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            
            transformed_pil = self.transform(img_pil)
            transformed_np = np.array(transformed_pil)
            
            return transformed_np
        
        elif isinstance(img, Image.Image):
            # Direct PIL Image
            return self.transform(img)
        
        else:
            raise TypeError(f"Unsupported input type: {type(img)}")


# 便利函数：创建高级数据增强组合
def create_advanced_transforms(image_size: int = 224, p: float = 0.5):
    """
    Create advanced data augmentation pipeline with custom transforms
    
    Args:
        image_size (int): target image size
        p (float): probability for each transform
    
    Returns:
        dict: dictionary of transform pipelines
    """
    
    # Import Jittor transforms
    from jittor.transform import (
        Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, 
        RandomAffine, ColorJitter, ToTensor, ImageNormalize
    )
    
    # Custom transforms
    grid_distortion = JittorTransformWrapper(GridDistortion(num_steps=5, distort_limit=0.05, p=1.0))
    elastic_transform = JittorTransformWrapper(ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0))
    coarse_dropout = JittorTransformWrapper(CoarseDropout(
        max_holes=8, max_height=image_size // 20, max_width=image_size // 20,
        min_holes=5, fill_value=0, p=1.0
    ))
    
    # OneOf equivalent for geometric transforms
    geometric_transforms = RandomChoice([grid_distortion, elastic_transform], p=0.25)
    geometric_wrapper = JittorTransformWrapper(geometric_transforms)
    
    # Training transforms
    transform_train = Compose([
        Resize((image_size, image_size)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomAffine(degrees=10, translate=[0.1, 0.1], scale=[0.9, 1.1], shear=10, p=0.5),
        geometric_wrapper,  # Advanced geometric transforms
        JittorTransformWrapper(coarse_dropout),  # Coarse dropout
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.8),
        ToTensor(),
        ImageNormalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                      std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Validation transforms
    transform_val = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        ImageNormalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                      std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    return {
        'train': transform_train,
        'val': transform_val
    }


# 测试和可视化函数
def visualize_transforms_with_real_image(debug_dir='/root/workspace/Jittor/Mycode2/debug'):
    """
    使用真实图像测试自定义变换并保存可视化结果
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from datetime import datetime
    
    # 创建debug目录
    os.makedirs(debug_dir, exist_ok=True)
    
    # 创建测试图像 - 使用更复杂的图案以便观察变换效果
    test_img = Image.new('RGB', (224, 224), color='white')
    
    # 添加网格图案以便更好地观察变换效果
    img_array = np.array(test_img)
    
    # 绘制网格和图案
    for i in range(0, 224, 28):  # 垂直线
        img_array[:, i:i+2, :] = [255, 0, 0]  # 红色线条
    for i in range(0, 224, 28):  # 水平线
        img_array[i:i+2, :, :] = [0, 255, 0]  # 绿色线条
    
    # 添加一些几何形状
    # 添加圆形
    y, x = np.ogrid[:224, :224]
    center_y, center_x = 80, 80
    mask = (x - center_x)**2 + (y - center_y)**2 <= 30**2
    img_array[mask] = [0, 0, 255]  # 蓝色圆形
    
    # 添加矩形
    img_array[120:170, 120:170, :] = [255, 255, 0]  # 黄色矩形
    
    test_img = Image.fromarray(img_array)
    
    # 设置matplotlib参数
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Custom Jittor Transform Visualization', fontsize=16, fontweight='bold')
    
    # 原始图像
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title('Original Image\n(Grid Pattern + Shapes)', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 测试GridDistortion
    print("Testing GridDistortion...")
    grid_transform = GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)
    distorted_img = grid_transform(test_img)
    axes[0, 1].imshow(distorted_img)
    axes[0, 1].set_title('GridDistortion\n(num_steps=5, distort_limit=0.3)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # 测试ElasticTransform
    print("Testing ElasticTransform...")
    elastic_transform = ElasticTransform(alpha=50, sigma=5, p=1.0)
    elastic_img = elastic_transform(test_img)
    axes[0, 2].imshow(elastic_img)
    axes[0, 2].set_title('ElasticTransform\n(alpha=50, sigma=5)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # 测试CoarseDropout
    print("Testing CoarseDropout...")
    dropout_transform = CoarseDropout(max_holes=10, max_height=20, max_width=20, p=1.0)
    dropout_img = dropout_transform(test_img)
    axes[1, 0].imshow(dropout_img)
    axes[1, 0].set_title('CoarseDropout\n(max_holes=10, hole_size=20x20)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # 测试RandomChoice (OneOf equivalent)
    print("Testing RandomChoice...")
    choice_transforms = RandomChoice([
        GridDistortion(num_steps=4, distort_limit=0.2, p=1.0),
        ElasticTransform(alpha=30, sigma=8, p=1.0)
    ], p=1.0)
    choice_img = choice_transforms(test_img)
    axes[1, 1].imshow(choice_img)
    axes[1, 1].set_title('RandomChoice\n(GridDistortion OR ElasticTransform)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # 测试组合变换
    print("Testing combined transforms...")
    combined_transforms = RandomApply([
        GridDistortion(num_steps=6, distort_limit=0.1, p=1.0),
        CoarseDropout(max_holes=5, max_height=15, max_width=15, p=1.0)
    ], p=1.0)
    combined_img = combined_transforms(test_img)
    axes[1, 2].imshow(combined_img)
    axes[1, 2].set_title('Combined Transforms\n(GridDistortion + CoarseDropout)', fontweight='bold')
    axes[1, 2].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存可视化结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_file = os.path.join(debug_dir, f'transform_visualization_{timestamp}.png')
    plt.savefig(viz_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {viz_file}")
    
    # 单独保存每个变换的结果
    individual_dir = os.path.join(debug_dir, f'individual_transforms_{timestamp}')
    os.makedirs(individual_dir, exist_ok=True)
    
    # 保存单独的图像
    test_img.save(os.path.join(individual_dir, 'original.png'))
    distorted_img.save(os.path.join(individual_dir, 'grid_distortion.png'))
    elastic_img.save(os.path.join(individual_dir, 'elastic_transform.png'))
    dropout_img.save(os.path.join(individual_dir, 'coarse_dropout.png'))
    choice_img.save(os.path.join(individual_dir, 'random_choice.png'))
    combined_img.save(os.path.join(individual_dir, 'combined_transforms.png'))
    
    print(f"Individual transform results saved to: {individual_dir}")
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    return viz_file, individual_dir


def test_jittor_compatibility(debug_dir='/root/workspace/Jittor/Mycode2/debug'):
    """
    测试JittorTransformWrapper的兼容性
    """
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    print("Testing Jittor compatibility...")
    
    # 创建测试图像
    test_img = Image.new('RGB', (64, 64), color='red')
    
    # 测试JittorTransformWrapper
    coarse_dropout = CoarseDropout(max_holes=3, max_height=8, max_width=8, p=1.0)
    wrapper = JittorTransformWrapper(coarse_dropout)
    
    # 测试不同输入类型
    results = {}
    
    # 1. PIL Image输入
    results['pil_input'] = wrapper(test_img)
    
    # 2. Numpy array输入 (HWC format)
    numpy_img = np.array(test_img)
    results['numpy_input'] = wrapper(numpy_img)
    
    # 3. Jittor tensor输入 (CHW format)
    jittor_img = jt.array(numpy_img.transpose(2, 0, 1).astype(np.float32) / 255.0)
    results['jittor_input'] = wrapper(jittor_img)
    
    # 可视化结果
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('JittorTransformWrapper Compatibility Test', fontsize=14, fontweight='bold')
    
    # 原始图像
    axes[0].imshow(test_img)
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # PIL输入结果
    if isinstance(results['pil_input'], Image.Image):
        axes[1].imshow(results['pil_input'])
    else:
        axes[1].imshow(np.array(results['pil_input']))
    axes[1].set_title('PIL Input → PIL Output', fontweight='bold')
    axes[1].axis('off')
    
    # Numpy输入结果
    axes[2].imshow(results['numpy_input'])
    axes[2].set_title('Numpy Input → Numpy Output', fontweight='bold')
    axes[2].axis('off')
    
    # Jittor输入结果
    jittor_result = results['jittor_input']
    if isinstance(jittor_result, jt.Var):
        jittor_result = jittor_result.numpy()
        if jittor_result.ndim == 3 and jittor_result.shape[0] in [1, 3]:
            jittor_result = jittor_result.transpose(1, 2, 0)
        jittor_result = np.clip(jittor_result, 0, 1)
    axes[3].imshow(jittor_result)
    axes[3].set_title('Jittor Input → Jittor Output', fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compat_file = os.path.join(debug_dir, f'jittor_compatibility_test_{timestamp}.png')
    plt.savefig(compat_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Compatibility test saved to: {compat_file}")
    
    plt.close(fig)
    
    print("Jittor compatibility test completed successfully!")
    return compat_file


# 使用示例
if __name__ == "__main__":
    print("Testing custom Jittor transforms...")
    print("="*60)
    
    # 基本功能测试
    print("1. Basic functionality test:")
    
    # 创建测试图像
    test_img = Image.new('RGB', (224, 224), color='red')
    
    # 测试GridDistortion
    grid_transform = GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)
    distorted_img = grid_transform(test_img)
    
    # 测试ElasticTransform
    elastic_transform = ElasticTransform(alpha=50, sigma=5, p=1.0)
    elastic_img = elastic_transform(test_img)
    
    # 测试CoarseDropout
    dropout_transform = CoarseDropout(max_holes=10, max_height=20, max_width=20, p=1.0)
    dropout_img = dropout_transform(test_img)
    
    print("✓ GridDistortion, ElasticTransform, and CoarseDropout created successfully!")
    
    # 可视化测试
    print("\n2. Visualization test:")
    try:
        viz_file, individual_dir = visualize_transforms_with_real_image()
        print("✓ Visualization test completed successfully!")
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
    
    # Jittor兼容性测试
    print("\n3. Jittor compatibility test:")
    try:
        compat_file = test_jittor_compatibility()
        print("✓ Jittor compatibility test completed successfully!")
    except Exception as e:
        print(f"✗ Jittor compatibility test failed: {e}")
    
    print("\n" + "="*60)
    print("All tests completed! Check /root/workspace/Jittor/Mycode2/debug for visualization results.")
    print("="*60)


# 完整的build_transform函数，支持自定义变换和标准Jittor变换
def build_transform(transform_cfg):
    """
    构建变换管道，支持标准Jittor变换和自定义变换
    
    Args:
        transform_cfg (list): 变换配置列表
        
    Returns:
        Compose: Jittor变换组合
    """
    from jittor.transform import Compose
    
    transform_list = []
    
    for i, item in enumerate(transform_cfg):
        # 添加调试信息
        print(f"Processing transform {i}: {item}")
        
        if not isinstance(item, dict):
            raise ValueError(f"Transform config item {i} must be a dictionary, got {type(item)}: {item}")
        
        if 'type' not in item:
            raise ValueError(f"Transform config item {i} missing 'type' field: {item}")
            
        t_type = item['type']
        params = {k: v for k, v in item.items() if k != 'type'}
        
        # 自动将 size/mean/std 等 list 转 tuple
        for key in ['size', 'mean', 'std']:
            if key in params and isinstance(params[key], list):
                params[key] = tuple(params[key])
        
        # 处理插值模式参数
        if 'mode' in params:
            if params['mode'] == 'BICUBIC':
                params['mode'] = Image.BICUBIC
            elif params['mode'] == 'BILINEAR':
                params['mode'] = Image.BILINEAR
            elif params['mode'] == 'NEAREST':
                params['mode'] = Image.NEAREST
        
        # 处理自定义变换
        if t_type in ['GridDistortion', 'ElasticTransform', 'CoarseDropout']:
            # 创建自定义变换并用JittorTransformWrapper包装
            if t_type == 'GridDistortion':
                custom_transform = GridDistortion(**params)
            elif t_type == 'ElasticTransform':
                custom_transform = ElasticTransform(**params)
            elif t_type == 'CoarseDropout':
                custom_transform = CoarseDropout(**params)
            
            wrapped_transform = JittorTransformWrapper(custom_transform)
            transform_list.append(wrapped_transform)
            
        elif t_type == 'RandomChoice':
            # 处理RandomChoice变换
            sub_transforms = []
            for sub_item in params.get('transforms', []):
                sub_type = sub_item['type']
                sub_params = {k: v for k, v in sub_item.items() if k != 'type'}
                
                # 处理子变换的参数
                for key in ['size', 'mean', 'std']:
                    if key in sub_params and isinstance(sub_params[key], list):
                        sub_params[key] = tuple(sub_params[key])
                
                if sub_type in ['GridDistortion', 'ElasticTransform', 'CoarseDropout']:
                    if sub_type == 'GridDistortion':
                        sub_transform = GridDistortion(**sub_params)
                    elif sub_type == 'ElasticTransform':
                        sub_transform = ElasticTransform(**sub_params)
                    elif sub_type == 'CoarseDropout':
                        sub_transform = CoarseDropout(**sub_params)
                    sub_transforms.append(sub_transform)
                else:
                    # 标准Jittor变换，需要动态导入
                    sub_transform = _create_jittor_transform(sub_type, sub_params)
                    sub_transforms.append(sub_transform)
            
            # 创建RandomChoice并包装
            choice_transform = RandomChoice(sub_transforms, p=params.get('p', 0.5))
            wrapped_transform = JittorTransformWrapper(choice_transform)
            transform_list.append(wrapped_transform)
            
        elif t_type == 'RandomApply':
            # 处理RandomApply变换
            sub_transforms = []
            for sub_item in params.get('transforms', []):
                sub_type = sub_item['type']
                sub_params = {k: v for k, v in sub_item.items() if k != 'type'}
                
                # 处理子变换的参数
                for key in ['size', 'mean', 'std']:
                    if key in sub_params and isinstance(sub_params[key], list):
                        sub_params[key] = tuple(sub_params[key])
                
                if sub_type in ['GridDistortion', 'ElasticTransform', 'CoarseDropout']:
                    if sub_type == 'GridDistortion':
                        sub_transform = GridDistortion(**sub_params)
                    elif sub_type == 'ElasticTransform':
                        sub_transform = ElasticTransform(**sub_params)
                    elif sub_type == 'CoarseDropout':
                        sub_transform = CoarseDropout(**sub_params)
                    sub_transforms.append(sub_transform)
                else:
                    # 标准Jittor变换，需要动态导入
                    sub_transform = _create_jittor_transform(sub_type, sub_params)
                    sub_transforms.append(sub_transform)
            
            # 创建RandomApply并包装
            # 支持apply_all参数控制应用模式
            apply_all = params.get('apply_all', True)  # 默认为True保持向后兼容
            apply_transform = RandomApply(sub_transforms, p=params.get('p', 0.5), apply_all=apply_all)
            wrapped_transform = JittorTransformWrapper(apply_transform)
            transform_list.append(wrapped_transform)
            
        else:
            # 标准Jittor变换
            transform = _create_jittor_transform(t_type, params)
            transform_list.append(transform)
    
    return Compose(transform_list)


def _create_jittor_transform(t_type, params):
    """
    创建标准Jittor变换的辅助函数
    
    Args:
        t_type (str): 变换类型名
        params (dict): 变换参数
        
    Returns:
        transform: Jittor变换实例
    """
    from jittor.transform import (
        Resize, CenterCrop, RandomCrop, RandomRotation, RandomVerticalFlip, 
        RandomHorizontalFlip, ToTensor, ImageNormalize, RandomResizedCrop, 
        RandomAffine, ColorJitter, RandomGray
    )
    
    # 变换类映射
    transform_map = {
        'Resize': Resize,
        'CenterCrop': CenterCrop,
        'RandomCrop': RandomCrop,
        'RandomRotation': RandomRotation,
        'RandomVerticalFlip': RandomVerticalFlip,
        'RandomHorizontalFlip': RandomHorizontalFlip,
        'ToTensor': ToTensor,
        'ImageNormalize': ImageNormalize,
        'RandomResizedCrop': RandomResizedCrop,
        'RandomAffine': RandomAffine,
        'ColorJitter': ColorJitter,
        'RandomGray': RandomGray,
    }
    
    if t_type in transform_map:
        transform_cls = transform_map[t_type]
        
        # 过滤掉标准Jittor变换不支持的参数
        # 大多数Jittor变换不支持'p'参数，需要过滤掉
        filtered_params = params.copy()
        
        # 定义支持'p'参数的Jittor变换
        transforms_with_p = {
            'RandomVerticalFlip', 'RandomHorizontalFlip', 'ColorJitter', 
            'RandomGray', 'RandomRotation', 'RandomAffine'
        }
        
        # 如果变换不支持'p'参数，则过滤掉
        if t_type not in transforms_with_p and 'p' in filtered_params:
            filtered_params.pop('p')
        
        return transform_cls(**filtered_params)
    else:
        raise ValueError(f"Unknown transform type: {t_type}")


# 便利函数：从YAML文件加载并构建变换
def build_transform_from_yaml(yaml_path, split='train'):
    """
    从YAML文件加载并构建变换管道
    
    Args:
        yaml_path (str): YAML配置文件路径
        split (str): 'train' 或 'val'
        
    Returns:
        Compose: Jittor变换组合
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        transform_config = yaml.load(f, Loader=yaml.FullLoader)
    
    if split == 'train':
        return build_transform(transform_config['transform_train'])
    elif split == 'val':
        return build_transform(transform_config['transform_val'])
    else:
        raise ValueError(f"Unknown split: {split}")


# 使用示例和配置模板
EXAMPLE_CONFIG = {
    'transform_train': [
        # 基础变换
        {'type': 'RandomResizedCrop', 'size': [224, 224], 'scale': [0.8, 1.0]},
        {'type': 'RandomHorizontalFlip', 'p': 0.5},
        
        # 自定义变换
        {'type': 'GridDistortion', 'num_steps': 5, 'distort_limit': 0.05, 'p': 0.3},
        {'type': 'ElasticTransform', 'alpha': 1, 'sigma': 50, 'p': 0.25},
        {'type': 'CoarseDropout', 'max_holes': 8, 'max_height': 11, 'max_width': 11, 'p': 0.3},
        
        # 概率应用单个变换 (新功能)
        {
            'type': 'RandomApply',
            'p': 0.5,
            'apply_all': False,  # 每个变换独立应用
            'transforms': [
                {'type': 'RandomAffine', 'degrees': 10, 'translate': [0.1, 0.1], 'p': 1.0}
            ]
        },
        
        # 随机选择变换
        {
            'type': 'RandomChoice',
            'p': 0.4,
            'transforms': [
                {'type': 'GridDistortion', 'num_steps': 4, 'distort_limit': 0.08, 'p': 1.0},
                {'type': 'ElasticTransform', 'alpha': 2, 'sigma': 40, 'p': 1.0}
            ]
        },
        
        # 标准变换
        {'type': 'ColorJitter', 'brightness': 0.1, 'contrast': 0.1, 'p': 0.8},
        {'type': 'ToTensor'},
        {'type': 'ImageNormalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    ],
    'transform_val': [
        {'type': 'Resize', 'size': [224, 224]},
        {'type': 'ToTensor'},
        {'type': 'ImageNormalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    ]
}
