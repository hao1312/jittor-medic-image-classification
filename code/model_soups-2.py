import os
import os.path as osp
import argparse
import yaml
from typing import List
from tqdm import tqdm

import jittor as jt
import jittor.nn as nn
from jimm import swin_base_patch4_window12_384

from utils.dataset import ImageFolder2, load_train_val_samples
from utils.val_utils import evaluate_val_set
from utils.jittor_transform import build_transform

jt.flags.use_cuda = 1


def uniform_soup(model_paths: List[str], num_classes: int = 6) -> nn.Module:
    """
    Uniform soup: 对所有模型参数进行等权重平均
    Args:
        model_paths: 模型文件路径列表
        num_classes: 分类数
    Returns:
        平均后的模型
    """
    print(f"Creating uniform soup from {len(model_paths)} models...")
    
    # 初始化模型
    soup_model = swin_base_patch4_window12_384(pretrained=False, num_classes=num_classes)
    
    # 加载第一个模型作为基础
    first_model = swin_base_patch4_window12_384(pretrained=False, num_classes=num_classes)
    first_model.load(model_paths[0])
    
    # 初始化soup模型参数
    soup_state_dict = {}
    for name, param in first_model.named_parameters():
        soup_state_dict[name] = param.data.copy()  # 使用copy()替代clone()
    
    # 累加其他模型的参数
    for model_path in tqdm(model_paths[1:], desc="Loading models"):
        model = swin_base_patch4_window12_384(pretrained=False, num_classes=num_classes)
        model.load(model_path)
        
        for name, param in model.named_parameters():
            soup_state_dict[name] += param.data
    
    # 计算平均值
    for name in soup_state_dict:
        soup_state_dict[name] /= len(model_paths)
    
    # 将平均参数加载到soup模型中
    for name, param in soup_model.named_parameters():
        param.assign(soup_state_dict[name])
    
    return soup_model


def collect_fold_models(results_path: str, model_type: str = 'best') -> List[str]:
    """
    收集所有fold的指定类型模型路径
    Args:
        results_path: 结果根目录路径
        model_type: 模型类型 ('best', 'best_ema', 'last', 'last_ema')
    Returns:
        模型路径列表
    """
    model_paths = []
    
    # 遍历所有fold目录
    for fold_dir in sorted(os.listdir(results_path)):
        if fold_dir.startswith('fold_'):
            fold_path = osp.join(results_path, fold_dir)
            model_path = osp.join(fold_path, f'{model_type}.pkl')
            
            if osp.exists(model_path):
                model_paths.append(model_path)
                print(f"Found model: {model_path}")
            else:
                print(f"Warning: Model not found: {model_path}")
    
    return model_paths


def main():
    parser = argparse.ArgumentParser(description='Model Soups for K-fold Cross Validation Results')
    parser.add_argument('--results_path', type=str, 
                        default='/root/workspace/Jittor/Mycode2/resultsv3/swinvit_384_imgnet_enhrans_lr1e4_warm5_rwldam_b22_noohem/',
                        help='Path to fold results directory')
    parser.add_argument('--model_type', type=str, default='best_ema',
                        choices=['best', 'best_ema', 'last', 'last_ema'],
                        help='Type of model to use for soup')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes')
    args = parser.parse_args()
    
    print(f"Model Soups for results in: {args.results_path}")
    
    # 收集模型路径
    model_paths = collect_fold_models(args.results_path, args.model_type)
    
    if len(model_paths) == 0:
        print("No models found!")
        return
    
    print(f"Found {len(model_paths)} models for soup creation.")
    
    # 创建Uniform Soup
    print("\n" + "="*50)
    print("Creating Uniform Soup...")
    print("="*50)
    
    uniform_soup_model = uniform_soup(model_paths, args.num_classes)
    
    # 保存uniform soup
    uniform_soup_model.save(osp.join(args.results_path, 'uniform_soup.pkl'))
    

if __name__ == "__main__":
    main()