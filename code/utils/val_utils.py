import os
import sys
import glob
import jittor as jt
import numpy as np
import yaml
import argparse
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from jittor.transform import (
    Compose, Resize, CenterCrop, ToTensor, ImageNormalize, RandomResizedCrop, 
    RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, ColorJitter
)
from jimm import swin_base_patch4_window12_384
from utils.dataset import load_train_val_samples, ImageFolder2
from sklearn.metrics import classification_report, confusion_matrix

jt.flags.use_cuda = 1

def plot_confusion_matrix(cm, class_names, save_path, normalize=True, save_type="normalized", epoch=None):
    """
    绘制并保存混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # 处理除零的情况
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        plt.title(f'Normalized Confusion Matrix')
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # 保存图片到images文件夹
    images_dir = os.path.join(save_path, 'val_images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 使用epoch信息和save_type命名文件
    if epoch is not None:
        save_file = os.path.join(images_dir, f'confusion_matrix_{save_type}_epoch_{epoch}.png')
    else:
        save_file = os.path.join(images_dir, f'confusion_matrix_{save_type}.png')
    
    plt.savefig(save_file, dpi=96, bbox_inches='tight')
    plt.close()
    
    return save_file


def evaluate_val_set_multi(models_dict, val_loader, num_classes=6, class_names=None, save_path=None, epoch=None):
    """
    评估多个模型并返回详细结果

    Args:
        models_dict: 字典，键为模型名称，值为模型对象，例如 {"original": model, "ema": ema_model}
        val_loader: 验证数据加载器
        num_classes: 类别数量
        class_names: 类别名称列表
        save_path: 保存路径
        epoch: 当前epoch
    
    Returns:
        results_dict: 字典，键为模型名称，值为该模型的结果元组 (micro_acc, macro_acc, report, cm)
    """
    # 将所有模型设置为评估模式
    for model_name, model in models_dict.items():
        model.eval()
    
    # 存储所有模型的预测结果
    all_preds = {name: [] for name in models_dict.keys()}
    targets = []

    # 改进显示信息
    if epoch == "final":
        eval_desc = "Final validation (multi-model)"
    else:
        eval_desc = "Evaluating validation set (multi-model)"

    print(f"{eval_desc}...")
    
    # 单次遍历数据，同时预测所有模型
    for batch_data in tqdm(val_loader, desc=eval_desc):
        if len(batch_data) == 2:  # (image, label)
            image, label = batch_data
        elif len(batch_data) == 3:  # (image, label, name)
            image, label, _ = batch_data
        else:
            raise ValueError("Batch data must contain 2 or 3 elements (image, label[, name])")
        
        # 对每个模型进行预测
        for model_name, model in models_dict.items():
            pred = model(image)
            pred.sync()
            all_preds[model_name].append(pred.numpy().argmax(axis=1))
        
        targets.append(label.numpy())
    
    # 合并目标标签
    y_true = np.concatenate(targets)
    
    # 如果没有提供类别名称，使用默认的
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # 处理每个模型的结果
    results_dict = {}
    
    for model_name in models_dict.keys():
        # 合并该模型的预测结果
        y_pred = np.concatenate(all_preds[model_name])
        
        # 生成详细的分类报告
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            labels=list(range(num_classes)),
            zero_division=0,
            output_dict=True
        )
        
        # 从report中提取指标
        macro_acc = report['macro avg']['precision']
        micro_acc = report['accuracy']
        
        # 提取每类准确率
        class_acc = []
        class_counts = []
        for i in range(num_classes):
            class_name = class_names[i]
            if class_name in report:
                class_acc.append(report[class_name]['recall'])
                class_counts.append(int(report[class_name]['support']))
            else:
                class_acc.append(float('nan'))
                class_counts.append(0)
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

        # 保存结果到文件
        if save_path is not None and epoch is not None:
            suffix = f"_{model_name}" if model_name != "original" else ""
            save_results_to_files(
                y_pred, y_true, macro_acc, micro_acc, report, cm, 
                class_names, class_acc, class_counts, 
                save_path, epoch, suffix
            )
        
        # 存储该模型的结果
        results_dict[model_name] = (micro_acc, macro_acc, report, cm)
    
    return results_dict


def evaluate_val_set(model, val_loader, num_classes=6, class_names=None, save_path=None, epoch=None, suffix=""):
    """评估验证集并返回详细结果（保持向后兼容）"""
    model.eval()
    preds, targets = [], []
    
    # 改进显示信息
    if epoch == "final":
        eval_desc = f"Final validation{suffix}" if suffix else "Final validation"
    else:
        eval_desc = f"Evaluating{suffix}" if suffix else "Evaluating validation set"
    
    print(f"{eval_desc}...")
    
    for batch_data in tqdm(val_loader, desc=eval_desc):
        if len(batch_data) == 2:  # (image, label)
            image, label = batch_data
        elif len(batch_data) == 3:  # (image, label, name)
            image, label, _ = batch_data
        else:
            raise ValueError("Batch data must contain 2 or 3 elements (image, label[, name])")
        
        pred = model(image)
        pred.sync()
        preds.append(pred.numpy().argmax(axis=1))
        targets.append(label.numpy())
    
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    
    # 如果没有提供类别名称，使用默认的
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # 生成详细的分类报告
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        labels=list(range(num_classes)),
        zero_division=0,
        output_dict=True
    )
    
    # 从report中提取指标
    macro_acc = report['macro avg']['precision']  # 使用macro avg precision作为macro accuracy
    micro_acc = report['accuracy']  # 使用overall accuracy作为micro accuracy
    
    # 提取每类准确率（从report中的recall获取）
    class_acc = []
    class_counts = []
    for i in range(num_classes):
        class_name = class_names[i]  # 使用实际的类别名称作为键
        if class_name in report:
            class_acc.append(report[class_name]['recall'])  # recall就是该类别的准确率
            class_counts.append(int(report[class_name]['support']))
        else:
            class_acc.append(float('nan'))
            class_counts.append(0)
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 保存结果到文件
    if save_path is not None and epoch is not None:
        save_results_to_files(
            y_pred, y_true, macro_acc, micro_acc, report, cm, 
            class_names, class_acc, class_counts, 
            save_path, epoch, suffix
        )
    
    # 返回简化的4个主要指标，与训练脚本的期望一致
    return micro_acc, macro_acc, report, cm


def save_results_to_files(y_pred, y_true, macro_acc, micro_acc, report, cm, 
                         class_names, class_acc, class_counts, 
                         save_path, epoch, suffix=""):
    """保存验证结果到文件"""
    
    # 创建保存目录
    images_dir = os.path.join(save_path, 'val_images')
    txt_dir = os.path.join(save_path, 'val_txt')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    # 保存混淆矩阵图片
    plot_confusion_matrix(cm, class_names, save_path, normalize=True, 
                         save_type=f"normalized{suffix}", epoch=epoch)
    plot_confusion_matrix(cm, class_names, save_path, normalize=False, 
                         save_type=f"raw{suffix}", epoch=epoch)
    
    # 保存详细报告到txt文件
    report_file = os.path.join(txt_dir, f'classification_report_epoch_{epoch}{suffix}.txt')
    with open(report_file, 'w') as f:
        f.write(f"Epoch {epoch} Validation Results{suffix}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Overall Accuracy (Micro): {micro_acc:.4f}\n")
        f.write(f"Macro Average Accuracy: {macro_acc:.4f}\n\n")
        
        f.write("Per-class Results:\n")
        f.write("-" * 30 + "\n")
        for i, (class_name, acc, count) in enumerate(zip(class_names, class_acc, class_counts)):
            f.write(f"{class_name}: Accuracy={acc:.4f}, Count={count}\n")
        
        f.write(f"\nTotal samples: {sum(class_counts)}\n\n")
        
        # 保存详细的分类报告
        f.write("Detailed Classification Report:\n")
        f.write("-" * 40 + "\n")
        report_str = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            labels=list(range(len(class_names))),
            zero_division=0
        )
        f.write(report_str)
        
        f.write(f"\n\nConfusion Matrix:\n")
        f.write("-" * 20 + "\n")
        f.write(f"True\\Pred\t" + "\t".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]}\t" + "\t".join(map(str, row)) + "\n")
    
    print(f"Results{suffix} saved to {report_file}")
    
    # 对于final验证，额外输出一些总结信息
    if str(epoch) == "final":
        print(f"Final validation completed{suffix}: Accuracy={micro_acc:.4f}, Macro Accuracy={macro_acc:.4f}")

