import argparse
import logging
import os
import os.path as osp
import sys
import yaml
import glob
import re
from tqdm import tqdm

import jittor as jt
import jittor.nn as nn
import numpy as np
from PIL import Image

from jittor.transform import (
    Compose, Resize, CenterCrop, ToTensor, ImageNormalize
)
from jimm import swin_base_patch4_window12_384

from utils.util import update_values

jt.flags.use_cuda = 1

def setup_logger(logfile):
    """
    设置日志记录器
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 移除所有旧的 handler
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    # 文件日志
    file_handler = logging.FileHandler(logfile, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    # 终端日志
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(stream_handler)


def extract_num(name):
    """
    从文件名中提取数字用于排序
    """
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else name


def inference(args, snapshot_path):
    """
    执行推理过程
    """
    # 1. 加载模型
    logging.info("正在加载Swin Transformer模型...")
    model = swin_base_patch4_window12_384(pretrained=False, num_classes=6)
    
    # 2. 加载权重
    weight_name = args.get('weight_name', 'best.pkl')
    weight_path = os.path.join(snapshot_path, weight_name)
    print(f"指定的权重文件: {weight_path}")
    
    # 如果指定的权重文件不存在，尝试其他常见名称
    if not os.path.exists(weight_path):
        fallback_weights = ['best.pkl', 'last.pkl', 'final.pkl']
        for fallback_weight in fallback_weights:
            fallback_path = os.path.join(snapshot_path, fallback_weight)
            if os.path.exists(fallback_path):
                weight_path = fallback_path
                logging.info(f"指定的权重文件 {weight_name} 不存在，使用 {fallback_weight}")
                break
        else:
            raise FileNotFoundError(f"在 {snapshot_path} 中未找到任何权重文件")
    
    logging.info(f"加载权重文件: {weight_path}")
    model.load(weight_path)
    model.eval()
    
    # 3. 构建transform
    logging.info("构建数据预处理...")
    # 使用自定义的384x384 transform
    transform_val = Compose([
        Resize((384, 384)),
        ToTensor(),
    ])
    logging.info("使用自定义的384x384 transform")
    
    # 4. 获取测试图片路径
    test_dir = args['test_dir']
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    
    if not img_paths:
        raise FileNotFoundError(f"在 {test_dir} 中未找到任何图片文件")
    
    img_paths = sorted(img_paths)
    logging.info(f"找到 {len(img_paths)} 张测试图片")
    
    # 5. 批量推理
    batch_size = args.get('batch_size', 16)
    results = []
    
    logging.info("开始推理...")
    for i in tqdm(range(0, len(img_paths), batch_size), desc="推理进度"):
        batch_paths = img_paths[i:i+batch_size]
        batch_imgs = []
        batch_names = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = transform_val(img)  # 经过transform后已经是(C,H,W)格式的tensor，已归一化
                # 直接添加到batch中，无需额外处理
                batch_imgs.append(img)
                batch_names.append(os.path.basename(img_path))
            except Exception as e:
                logging.warning(f"处理图片 {img_path} 时出错: {e}")
                continue
        
        if not batch_imgs:
            continue
            
        # 将list of tensors转换为batch tensor
        batch_imgs = jt.stack(batch_imgs, dim=0)
        
        with jt.no_grad():
            pred = model(batch_imgs)
            pred.sync()
            pred_labels = np.argmax(pred.numpy(), axis=1)
        
        for name, pred_label in zip(batch_names, pred_labels):
            results.append((name, int(pred_label)))
            if args.get('verbose', False):
                print(f"{name}: {pred_label}")
    
    # 6. 保存结果
    logging.info("保存推理结果...")
    os.makedirs(snapshot_path, exist_ok=True)
    
    # 按文件名中的数字排序
    results_sorted = sorted(results, key=lambda x: extract_num(x[0]))
    
    result_file = os.path.join(snapshot_path, "result.txt")
    with open(result_file, "w", encoding='utf-8') as f:
        for name, pred in results_sorted:
            f.write(f"{name} {pred}\n")
    
    logging.info(f"推理完成！结果已保存到: {result_file}")
    logging.info(f"总共处理了 {len(results)} 张图片")
    
    return result_file


def main():
    parser = argparse.ArgumentParser(description='Swin Transformer推理脚本')
    
    # 必需参数
    parser.add_argument('--snapshot_path', type=str, default='/root/workspace/Jittor/Mycode2/resultsv3/swinvit_384_imgnet_enhrans_lr1e4_warm5_rwldam_b22_noohem/model_soups_best_ema',
                       help='训练结果保存路径，包含权重文件的目录')
    parser.add_argument('--test_dir', type=str, default='/root/workspace/jittor/ultrasound_grading/TestSetA',
                       help='测试图片目录路径')
    
    # 可选参数
    parser.add_argument('--weight_name', type=str, default='uniform_soup.pkl',
                       help='权重文件名 (默认: uniform_soup.pkl)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批处理大小 (默认: 16)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细的推理过程')
    parser.add_argument('--log_file', type=str, default=None,
                       help='日志文件路径 (默认保存到snapshot_path/inference.log)')
    
    args = parser.parse_args()
    args = vars(args)
    
    # 设置日志
    if not args['log_file']:
        args['log_file'] = os.path.join(args['snapshot_path'], 'inference.log')
    print(f"日志文件将保存到: {args['log_file']}")
    
    setup_logger(args['log_file'])
    
    # 记录参数
    logging.info("="*60)
    logging.info("SWIN TRANSFORMER 推理配置")
    logging.info("="*60)
    logging.info(f"Snapshot Path: {args['snapshot_path']}")
    logging.info(f"Test Directory: {args['test_dir']}")
    logging.info(f"Weight Name: {args['weight_name']}")
    logging.info(f"Batch Size: {args['batch_size']}")
    logging.info(f"Verbose: {args['verbose']}")
    logging.info("="*60)
    
    try:
        result_file = inference(args, args['snapshot_path'])
        logging.info("推理成功完成！")
        print(f"\n推理结果已保存到: {result_file}")
    except Exception as e:
        logging.error(f"推理过程中出现错误: {e}")
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
        
        