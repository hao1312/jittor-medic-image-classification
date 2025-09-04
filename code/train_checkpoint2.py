import argparse
import logging
import os
import os.path as osp
import sys
import yaml
import random
import math
import pprint

import numpy as np
from tqdm import tqdm

import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import (
    Compose, Resize, CenterCrop, RandomCrop, RandomRotation, RandomVerticalFlip, 
    RandomHorizontalFlip, ToTensor, ImageNormalize, RandomResizedCrop, RandomAffine, 
    ColorJitter
)

from utils.val_utils import evaluate_val_set, evaluate_val_set_multi
from utils.ema import EMA
from utils.dataset import load_train_val_samples, split_kfold_dataset, ImageFolder2
from utils.samplerv4 import BalanceImageFolder2, ManualBalancedSampler, EfficientBalancedSampler
from utils.ldamlossV2 import RWLDAMDRWLoss, OHEMSampler

from jimm import swin_base_patch4_window12_384

# 导入自定义变换和构建函数
from utils.jittor_transform import (
    GridDistortion, ElasticTransform, CoarseDropout, 
    JittorTransformWrapper, RandomChoice, RandomApply,
    build_transform  # 直接导入build_transform函数
)

jt.flags.use_cuda = 1

def compute_class_distribution(samples, num_classes):
    """
    统计训练样本的类别分布
    
    Args:
        samples: 样本列表，每个元素为 (image_path, label)
        num_classes: 类别总数
    
    Returns:
        list: 每个类别的样本数量，按类别标签 0,1,2,... 的顺序排列
    """
    class_counts = [0] * num_classes
    
    for _, label in samples:
        if 0 <= label < num_classes:
            class_counts[label] += 1
        else:
            logging.warning(f"发现超出范围的标签: {label}, 期望范围: 0-{num_classes-1}")
    
    return class_counts


def training(model: nn.Module, criterion, optimizer: nn.Optimizer,
             train_loader: Dataset, now_epoch: int, num_epochs: int,
             global_step: int, warmup_steps: int, base_lr: float, 
             total_steps: int, cosine: bool, final_lr: float, ema=None, 
             ohem_sampler=None, ohem_start_epoch=0):
    model.train()
    losses = []
    # 计算正确的batch数量
    total_batches = (len(train_loader) + train_loader.batch_size - 1) // train_loader.batch_size
    pbar = tqdm(train_loader, total=total_batches,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    def get_lr(global_step):
        if global_step < warmup_steps:
            return base_lr * global_step / warmup_steps
        if cosine:
            progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
            lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))
            return lr
        else:
            return base_lr
    
    for step, data in enumerate(pbar, 1):
        global_step += 1
        image, label = data
        
        lr = get_lr(global_step)
        for group in optimizer.param_groups:
            group['lr'] = lr

        pred = model(image)
        
        # 判断是否使用OHEM
        use_ohem = ohem_sampler is not None and now_epoch >= ohem_start_epoch
        
        if use_ohem:
            # 使用OHEM计算损失
            loss, selected_indices, selection_info = ohem_sampler.apply_ohem(
                criterion, pred, label, now_epoch
            )
            # 可选：记录OHEM选择信息
            if step == 1:  # 只在第一个batch记录
                logging.info(f"OHEM Epoch {now_epoch}: {selection_info['strategy']} strategy, "
                           f"selected {selection_info['selected_samples']}/{selection_info['total_samples']} samples")
        else:
            # 标准损失计算 - RWLDAMDRWLoss需要epoch参数
            loss = criterion(pred, label, now_epoch)
            
        loss.sync()
        optimizer.step(loss)
        
        # 更新EMA
        if ema is not None:
            ema.update(model)

        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss = {losses[-1]:.4f} lr={lr:.6f}')

    logging.info(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.4f} lr={lr:.6f}')
    return global_step

def run(args, snapshot_path):
    model = swin_base_patch4_window12_384(pretrained=True, num_classes=6)
    
    # 从外部yml读取transform配置
    with open(args['transform_cfg'], 'r') as f:
        transform_yaml = yaml.load(f, Loader=yaml.FullLoader)
    transform_train = build_transform(transform_yaml['transform_train'])
    transform_val = build_transform(transform_yaml['transform_val'])

    train_samples, val_samples = split_kfold_dataset(
        osp.join(args['root_path'], 'labels/trainval.txt'),
        fold=args['fold'], num_folds=args['total_folds'],
        shuffle=True, seed=args.get('seed', 42)
    )
    logging.info(f"Fold {args['fold']} / {args['total_folds']} - 使用样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

    # ===== 动态统计训练集标签分布 =====
    actual_cls_num_list = compute_class_distribution(train_samples, args.get('num_classes', 6))
    logging.info(f"实际训练集类别分布: {actual_cls_num_list}")

    # 根据实际的类别分布创建RW-LDAM-DRW损失函数
    criterion = RWLDAMDRWLoss(
        cls_num_list=actual_cls_num_list,
        max_m=args.get('max_m', 0.5),
        s=args.get('s', 30),
        reweight_epoch=args.get('reweight_epoch', 80),
        total_epochs=args['epochs'],
        reweight_type=args.get('reweight_type', 'inverse')
    )
    
    # 初始化OHEM采样器（如果启用）
    ohem_sampler = None
    if args.get('use_ohem', False):
        ohem_sampler = OHEMSampler(
            hard_ratio=args.get('ohem_hard_ratio', 0.7),
            min_kept=args.get('ohem_min_kept', 16),
            selection_strategy=args.get('ohem_strategy', 'loss')
        )
        logging.info(f"启用OHEM - 困难样本比例: {args.get('ohem_hard_ratio', 0.7)}, "
                    f"最少保留: {args.get('ohem_min_kept', 16)}, "
                    f"选择策略: {args.get('ohem_strategy', 'loss')}")
    
    logging.info(f"使用RW-LDAM-DRW损失函数")
    logging.info(f"实际类别分布: {actual_cls_num_list}")
    logging.info(f"重加权开始epoch: {args.get('reweight_epoch', 80)}")
    logging.info(f"LDAM边距参数 max_m: {args.get('max_m', 0.5)}")
    logging.info(f"LDAM缩放因子 s: {args.get('s', 30)}")
    logging.info(f"重加权类型: {args.get('reweight_type', 'inverse')}")

    # 使用手动指定的过采样系数
    class_oversample_ratios = args.get('class_oversample_ratios', {
        0: 1.0, 1: 1.0, 2: 1.5, 3: 8.0, 4: 12.0, 5: 1.0
    })
    train_loader = BalanceImageFolder2(
        root=os.path.join(args['root_path'], 'images/train'),
        samples=train_samples,
        transform=transform_train,
        use_balanced_sampling=True,
        class_oversample_ratios=class_oversample_ratios,
        batch_size=args['batch_size'],
        num_workers=8,
        shuffle=True
    )
    logging.info(f"使用手动平衡采样，过采样系数: {class_oversample_ratios}")

    val_loader = ImageFolder2(
        root=os.path.join(args['root_path'], 'images/train'),
        samples=val_samples,
        transform=transform_val,
        batch_size=args['batch_size'],
        num_workers=8,
        shuffle=False
    )

    optimizer = jt.optim.AdamW(model.parameters(), lr=args['base_lr'])
    
    # 初始化EMA
    ema = None
    if args.get('use_ema', True):
        ema_decay = args.get('ema_decay', 0.9999)
        ema = EMA(model, decay=ema_decay)
        logging.info(f"EMA initialized with decay={ema_decay}")
    
    best_acc = 0
    best_ema_acc = 0
    global_step = 0
    num_epochs = args['epochs']
    warmup_epochs = args.get('warmup_epochs', 5)
    cosine = args.get('cosine', False)
    final_lr = args.get('final_lr', 1e-6)
    steps_per_epoch = (len(train_loader) + train_loader.batch_size - 1) // train_loader.batch_size
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    base_lr = args['base_lr']

    # 定义类别名称（根据您的数据集调整）
    class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']  # 根据实际类别调整

    for epoch in range(num_epochs):
        global_step = training(
            model, criterion, optimizer,
            train_loader, epoch, num_epochs,
            global_step, warmup_steps, base_lr,
            total_steps, cosine, final_lr, ema, 
            ohem_sampler, args.get('ohem_start_epoch', 0)
        )
        
        # 准备验证模型字典
        models_to_eval = {"original": model}
        if ema is not None:
            models_to_eval["ema"] = ema.get_model()
        
        # 使用多模型评估函数，一次遍历完成所有模型验证
        results_dict = evaluate_val_set_multi(
            models_to_eval, val_loader, 
            num_classes=6, 
            class_names=class_names, 
            save_path=snapshot_path, 
            epoch=epoch
        )
        
        # 提取原始模型结果
        acc, macro_acc, report, cm = results_dict["original"]
        
        # 记录详细的验证指标
        logging.info(f"Epoch {epoch} [VAL] Original Model:")
        logging.info(f"  Overall Accuracy: {acc:.4f}")
        logging.info(f"  Macro Average: {macro_acc:.4f}")
        
        # 记录每个类别的准确率
        class_accs = []
        for i, class_name in enumerate(class_names):
            if class_name in report:
                class_acc = report[class_name]['recall']
                class_precision = report[class_name]['precision']
                class_f1 = report[class_name]['f1-score']
                class_support = report[class_name]['support']
                class_accs.append(class_acc)
                logging.info(f"  {class_name}: Acc={class_acc:.4f}, P={class_precision:.4f}, F1={class_f1:.4f}, N={class_support}")
            else:
                class_accs.append(0.0)
                logging.info(f"  {class_name}: No samples")
        
        # 处理EMA模型结果（如果存在）
        if ema is not None:
            ema_acc, ema_macro_acc, ema_report, ema_cm = results_dict["ema"]
            
            # 记录EMA模型的详细指标
            logging.info(f"Epoch {epoch} [VAL] EMA Model:")
            logging.info(f"  Overall Accuracy: {ema_acc:.4f}")
            logging.info(f"  Macro Average: {ema_macro_acc:.4f}")
            
            # 记录EMA模型每个类别的准确率
            ema_class_accs = []
            for i, class_name in enumerate(class_names):
                if class_name in ema_report:
                    ema_class_acc = ema_report[class_name]['recall']
                    ema_class_precision = ema_report[class_name]['precision']
                    ema_class_f1 = ema_report[class_name]['f1-score']
                    ema_class_support = ema_report[class_name]['support']
                    ema_class_accs.append(ema_class_acc)
                    logging.info(f"  {class_name}: Acc={ema_class_acc:.4f}, P={ema_class_precision:.4f}, F1={ema_class_f1:.4f}, N={ema_class_support}")
                else:
                    ema_class_accs.append(0.0)
                    logging.info(f"  {class_name}: No samples")
        else:
            ema_acc = 0
            ema_macro_acc = 0

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            model.save(os.path.join(snapshot_path, 'best.pkl'))
        
        if ema is not None and ema_acc > best_ema_acc:
            best_ema_acc = ema_acc
            ema.get_model().save(os.path.join(snapshot_path, 'best_ema.pkl'))
        
        # 保存最后一个epoch的模型
        if epoch == num_epochs - 1:
            model.save(os.path.join(snapshot_path, 'last.pkl'))
            if ema is not None:
                ema.get_model().save(os.path.join(snapshot_path, 'last_ema.pkl'))
        
        # 定期保存checkpoint
        if (epoch + 1) % 50 == 0:
            model.save(os.path.join(snapshot_path, f'epoch_{epoch + 1}.pkl'))
            if ema is not None:
                ema.get_model().save(os.path.join(snapshot_path, f'epoch_{epoch + 1}_ema.pkl'))

        # 日志记录
        if ema is not None:
            logging.info(f'Epoch {epoch} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}, '
                        f'ema_best_acc = {best_ema_acc:.2f}, ema_acc = {ema_acc:.2f}')
        else:
            logging.info(f'Epoch {epoch} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform_cfg', type=str, default='/root/workspace/Jittor/Mycode2/cfgs/transform1_224.yml', help='transform配置文件')
    parser.add_argument('--root_path', type=str, default='/root/workspace/Jittor/DATASET/TrainSet')
    parser.add_argument('--res_path', type=str, default='/root/workspace/Jittor/Mycode2/resultsv3/')
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--final_lr', type=float, default=1e-6, help='cosine退火的最小学习率')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup轮数')
    parser.add_argument('--cosine', action='store_true', help='使用cosine退火调度')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_interval_ep', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    
    # K折交叉验证参数
    parser.add_argument('--fold', type=int, default=0, help='当前fold编号 (0-based)')
    parser.add_argument('--total_folds', type=int, default=4, help='总fold数量')
    
    # 数据采样策略参数
    parser.add_argument('--class_oversample_ratios', type=str, default='0:1.0,1:1.0,2:1.5,3:7.5,4:12.0,5:1.0',
                       help='手动平衡采样的各类别过采样系数，格式: 0:1.0,1:1.0,2:1.5...')
    
    # RW-LDAM-DRW损失函数参数
    parser.add_argument('--max_m', type=float, default=0.5, help='LDAM损失的最大边距值')
    parser.add_argument('--s', type=float, default=30, help='RW-LDAM-DRW的缩放因子')
    parser.add_argument('--reweight_epoch', type=int, default=80, help='开始重加权的epoch')
    parser.add_argument('--reweight_type', type=str, default='inverse', choices=['inverse', 'sqrt_inverse'], 
                       help='重加权类型')
    
    # OHEM参数
    parser.add_argument('--use_ohem', action='store_true', help='启用OHEM')
    parser.add_argument('--ohem_start_epoch', type=int, default=0, help='OHEM开始的epoch')
    parser.add_argument('--ohem_hard_ratio', type=float, default=0.7, help='OHEM困难样本比例')
    parser.add_argument('--ohem_min_kept', type=int, default=16, help='OHEM最少保留样本数')
    parser.add_argument('--ohem_strategy', type=str, default='loss', 
                       choices=['loss', 'uncertainty', 'combined'],
                       help='OHEM选择策略')
    
    # EMA相关参数
    parser.add_argument('--use_ema', action='store_true', default=True, help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')

    args = parser.parse_args()
    args = vars(args)
    
    
    # 解析class_oversample_ratios字符串为字典
    if 'class_oversample_ratios' in args and isinstance(args['class_oversample_ratios'], str):
        ratio_dict = {}
        for pair in args['class_oversample_ratios'].split(','):
            cls, ratio = pair.split(':')
            ratio_dict[int(cls)] = float(ratio)
        args['class_oversample_ratios'] = ratio_dict
    
    snapshot_path = osp.join(args["res_path"], args["exp"], f'fold_{args["fold"]}')
    os.makedirs(snapshot_path, exist_ok=True)
    
    # 创建验证结果保存的目录结构
    images_dir = osp.join(snapshot_path, 'images')
    val_txt_dir = osp.join(snapshot_path, 'val_txt')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(val_txt_dir, exist_ok=True)
    
    if args.get('deterministic', False):
        seed = args.get('seed', 2023)
        jt.set_global_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        logging.info(f"设置随机种子: {seed}")

    # os.environ['http_proxy'] = 'http://127.0.0.1:20171'
    # os.environ['https_proxy'] = 'https://127.0.0.1:20171'


    # 设置日志配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 移除所有旧的 handler
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    
    # 文件日志
    file_handler = logging.FileHandler(osp.join(snapshot_path, "log.txt"), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    
    # 终端日志
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(stream_handler)

    import pprint
    logging.info(pprint.pformat(args))

    run(args, snapshot_path)

