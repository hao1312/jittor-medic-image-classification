import argparse
import logging
import os
import os.path as osp
import sys
import yaml
import random
import math

import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import (
    Compose, Resize, CenterCrop, RandomCrop, RandomRotation, RandomVerticalFlip, 
    RandomHorizontalFlip, ToTensor, ImageNormalize, RandomResizedCrop, RandomAffine, 
    ColorJitter
)

from utils.util import update_values
from utils.val_utils import evaluate_val_set
from utils.ema import EMA
from utils.dataset import ImageFolder2, load_train_val_samples, split_kfold_dataset
from utils.sampler import BalancedImageFolder2
from utils.sam import SAM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from jimm import swin_base_patch4_window12_384
from jimm.loss import LabelSmoothingCrossEntropy

# 导入自定义变换和构建函数
from utils.jittor_transform import (
    GridDistortion, ElasticTransform, CoarseDropout, 
    JittorTransformWrapper, RandomChoice, RandomApply,
    build_transform  # 直接导入build_transform函数
)

# 导入不平衡损失函数
from utils.imbalanced_loss import create_imbalanced_criterion

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

def analyze_class_imbalance(cls_num_list):
    """
    分析类别不平衡程度并给出建议
    
    Args:
        cls_num_list: 每个类别的样本数量列表
    
    Returns:
        dict: 包含分析结果和建议的字典
    """
    total_samples = sum(cls_num_list)
    max_samples = max(cls_num_list)
    min_samples = min(cls_num_list)
    mean_samples = total_samples / len(cls_num_list)
    
    if min_samples == 0:
        imbalance_ratio = float('inf')
        empty_classes = [i for i, count in enumerate(cls_num_list) if count == 0]
        logging.warning(f"发现空类别: {empty_classes}")
    else:
        imbalance_ratio = max_samples / min_samples
    
    # 计算变异系数 (CV)
    import math
    variance = sum((count - mean_samples) ** 2 for count in cls_num_list) / len(cls_num_list)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_samples if mean_samples > 0 else float('inf')
    
    # 分析不平衡程度
    if imbalance_ratio <= 3:
        balance_level = "轻微不平衡"
        recommendations = ["可以使用标准交叉熵损失", "考虑数据增强"]
    elif imbalance_ratio <= 10:
        balance_level = "中等不平衡"
        recommendations = ["建议使用Focal Loss", "考虑重采样策略", "使用类别权重"]
    elif imbalance_ratio <= 50:
        balance_level = "严重不平衡"
        recommendations = ["强烈建议使用LDAM-DRW", "使用Class-Balanced Loss", "结合多种重采样策略"]
    else:
        balance_level = "极度不平衡"
        recommendations = ["必须使用专门的不平衡学习方法", "考虑生成合成样本", "使用集成方法"]
    
    return {
        'total_samples': total_samples,
        'max_samples': max_samples,
        'min_samples': min_samples,
        'mean_samples': mean_samples,
        'imbalance_ratio': imbalance_ratio,
        'coefficient_of_variation': cv,
        'balance_level': balance_level,
        'recommendations': recommendations
    }

def training(model: nn.Module, criterion, optimizer: nn.Optimizer,
             train_loader: Dataset, now_epoch: int, num_epochs: int,
             global_step: int, warmup_steps: int, base_lr: float, 
             total_steps: int, cosine: bool, final_lr: float, ema=None, use_epoch_in_loss=False):
    model.train()
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader),
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
        
        # 根据损失函数类型决定是否传递epoch参数
        if use_epoch_in_loss:
            loss = criterion(pred, label, now_epoch)
        else:
            loss = criterion(pred, label)
            
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
    # 创建损失函数
    loss_type = args.get('loss_type', 'label_smoothing')
    
    if loss_type == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy()
        use_epoch_in_loss = False
        logging.info("使用标签平滑交叉熵损失")
    elif loss_type in ['rw_ldam_drw', 'ce_ldam_drw', 'ce_drw', 'focal', 'cb_focal', 'cb_ce', 'rw_ce']:
        # 获取类别分布 (必须按类别标签0,1,2,3,4,5...的顺序)
        cls_num_list = args.get('cls_num_list', [480, 800, 280, 30, 10, 400])  # 默认6类分布，仅为示例！
        
        # 构建损失函数参数
        criterion_kwargs = {}
        
        # 通用参数
        if loss_type in ['rw_ldam_drw', 'ce_ldam_drw', 'ce_drw']:
            criterion_kwargs['reweight_epoch'] = args.get('reweight_epoch', 80)
            criterion_kwargs['total_epochs'] = args['epochs']
            criterion_kwargs['reweight_type'] = args.get('reweight_type', 'inverse')
        
        # LDAM相关参数
        if loss_type in ['rw_ldam_drw', 'ce_ldam_drw']:
            criterion_kwargs['max_m'] = args.get('max_m', 0.5)
        
        # RW-LDAM-DRW特有的缩放因子
        if loss_type == 'rw_ldam_drw':
            criterion_kwargs['s'] = args.get('s', 30)
        
        # Focal Loss参数
        if loss_type == 'focal':
            criterion_kwargs['alpha'] = args.get('focal_alpha', 1.0)
            criterion_kwargs['gamma'] = args.get('focal_gamma', 2.0)
        
        # Class-Balanced Loss参数
        if loss_type in ['cb_focal', 'cb_ce']:
            criterion_kwargs['beta'] = args.get('cb_beta', 0.9999)
            if loss_type == 'cb_focal':
                criterion_kwargs['gamma'] = args.get('focal_gamma', 2.0)
        
        criterion = create_imbalanced_criterion(loss_type, cls_num_list, **criterion_kwargs)
        use_epoch_in_loss = loss_type in ['rw_ldam_drw', 'ce_ldam_drw', 'ce_drw']
        
        # 日志记录损失函数配置
        logging.info(f"使用损失函数: {loss_type}")
        logging.info(f"类别分布: {cls_num_list}")
        if 'reweight_epoch' in criterion_kwargs:
            logging.info(f"重加权开始epoch: {criterion_kwargs['reweight_epoch']}")
        if 'max_m' in criterion_kwargs:
            logging.info(f"LDAM边距参数 max_m: {criterion_kwargs['max_m']}")
        if 's' in criterion_kwargs:
            logging.info(f"LDAM缩放因子 s: {criterion_kwargs['s']}")
        if 'alpha' in criterion_kwargs:
            logging.info(f"Focal Loss alpha: {criterion_kwargs['alpha']}")
        if 'gamma' in criterion_kwargs:
            logging.info(f"Focal Loss gamma: {criterion_kwargs['gamma']}")
        if 'beta' in criterion_kwargs:
            logging.info(f"Class-Balanced beta: {criterion_kwargs['beta']}")
        if 'reweight_type' in criterion_kwargs:
            logging.info(f"重加权类型: {criterion_kwargs['reweight_type']}")
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

    # 从外部yml读取transform配置
    with open(args['transform_cfg'], 'r') as f:
        transform_yaml = yaml.load(f, Loader=yaml.FullLoader)
    transform_train = build_transform(transform_yaml['transform_train'])
    transform_val = build_transform(transform_yaml['transform_val'])

    # train_samples, val_samples = load_train_val_samples(args['root_path'])
    # print("used samples:", len(train_samples), len(val_samples))


    train_samples, val_samples = split_kfold_dataset(
        osp.join(args['root_path'], 'labels/trainval.txt'),
        fold=args['fold'], num_folds=args['total_folds'],
    )
    logging.info(f"Fold {args['fold']} / {args['total_folds']} - 使用样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

    train_loader = BalancedImageFolder2(
        root=os.path.join(args['root_path'], 'images/train'),
        samples=train_samples,
        transform=transform_train,
        batch_size=args['batch_size'],
        num_workers=8,
        shuffle=True,
        use_balanced_sampling=True, 
        minority_boost_ratio=args.get('minority_boost_ratio', 0.4)
    )

    val_loader = ImageFolder2(
        root=os.path.join(args['root_path'], 'images/train'),
        samples=val_samples,
        transform=transform_val,
        batch_size=args['batch_size'],
        num_workers=8,
        shuffle=False
    )

    # ===== 动态统计训练集标签分布 =====
    actual_cls_num_list = compute_class_distribution(train_samples, args.get('num_classes', 6))
    logging.info(f"实际训练集类别分布: {actual_cls_num_list}")
    
    # 深度分析类别不平衡程度
    imbalance_analysis = analyze_class_imbalance(actual_cls_num_list)
    
    logging.info("=" * 60)
    logging.info("训练集类别分布分析报告")
    logging.info("=" * 60)
    logging.info(f"总样本数: {imbalance_analysis['total_samples']}")
    logging.info(f"最大类样本数: {imbalance_analysis['max_samples']}")
    logging.info(f"最小类样本数: {imbalance_analysis['min_samples']}")
    logging.info(f"平均样本数: {imbalance_analysis['mean_samples']:.1f}")
    logging.info(f"不平衡比率 (max/min): {imbalance_analysis['imbalance_ratio']:.2f}")
    logging.info(f"变异系数 (CV): {imbalance_analysis['coefficient_of_variation']:.3f}")
    logging.info(f"不平衡程度: {imbalance_analysis['balance_level']}")
    
    # 输出每个类别的详细信息
    logging.info("-" * 40)
    for i, count in enumerate(actual_cls_num_list):
        percentage = (count / imbalance_analysis['total_samples']) * 100
        logging.info(f"类别 {i}: {count:4d} 样本 ({percentage:5.2f}%)")
    
    logging.info("-" * 40)
    logging.info("建议策略:")
    for rec in imbalance_analysis['recommendations']:
        logging.info(f"  • {rec}")
    logging.info("=" * 60)
    
    # 根据实际分布动态调整损失函数参数
    if loss_type in ['rw_ldam_drw', 'ce_ldam_drw', 'ce_drw', 'focal', 'cb_focal', 'cb_ce', 'rw_ce']:
        # 使用实际统计的类别分布重新创建损失函数
        logging.info("根据实际训练集分布重新配置损失函数...")
        
        # 重新构建损失函数参数，使用实际分布
        criterion_kwargs = {}
        
        # 通用参数
        if loss_type in ['rw_ldam_drw', 'ce_ldam_drw', 'ce_drw']:
            criterion_kwargs['reweight_epoch'] = args.get('reweight_epoch', 80)
            criterion_kwargs['total_epochs'] = args['epochs']
            criterion_kwargs['reweight_type'] = args.get('reweight_type', 'inverse')
        
        # 根据不平衡程度动态调整LDAM参数
        if loss_type in ['rw_ldam_drw', 'ce_ldam_drw']:
            # 根据不平衡比率动态调整max_m，更不平衡的数据集使用更大的边距
            imbalance_ratio = imbalance_analysis['imbalance_ratio']
            if imbalance_ratio <= 10:
                dynamic_max_m = 0.3
            elif imbalance_ratio <= 50:
                dynamic_max_m = 0.5
            elif imbalance_ratio <= 100:
                dynamic_max_m = 0.7
            else:
                dynamic_max_m = 1.0
            
            criterion_kwargs['max_m'] = args.get('max_m', dynamic_max_m)
            logging.info(f"根据不平衡比率 {imbalance_ratio:.2f} 动态调整 max_m = {criterion_kwargs['max_m']}")
        
        # RW-LDAM-DRW特有的缩放因子
        if loss_type == 'rw_ldam_drw':
            criterion_kwargs['s'] = args.get('s', 30)
        
        # Focal Loss参数
        if loss_type == 'focal':
            criterion_kwargs['alpha'] = args.get('focal_alpha', 1.0)
            criterion_kwargs['gamma'] = args.get('focal_gamma', 2.0)
        
        # Class-Balanced Loss参数
        if loss_type in ['cb_focal', 'cb_ce']:
            criterion_kwargs['beta'] = args.get('cb_beta', 0.9999)
            if loss_type == 'cb_focal':
                criterion_kwargs['gamma'] = args.get('focal_gamma', 2.0)
        
        # 使用实际类别分布重新创建损失函数
        criterion = create_imbalanced_criterion(loss_type, actual_cls_num_list, **criterion_kwargs)
        
        # 重新记录损失函数配置
        logging.info(f"重新配置损失函数: {loss_type}")
        logging.info(f"使用实际类别分布: {actual_cls_num_list}")
        if 'reweight_epoch' in criterion_kwargs:
            logging.info(f"重加权开始epoch: {criterion_kwargs['reweight_epoch']}")
        if 'max_m' in criterion_kwargs:
            logging.info(f"LDAM边距参数 max_m: {criterion_kwargs['max_m']}")
        if 's' in criterion_kwargs:
            logging.info(f"LDAM缩放因子 s: {criterion_kwargs['s']}")
        if 'alpha' in criterion_kwargs:
            logging.info(f"Focal Loss alpha: {criterion_kwargs['alpha']}")
        if 'gamma' in criterion_kwargs:
            logging.info(f"Focal Loss gamma: {criterion_kwargs['gamma']}")
        if 'beta' in criterion_kwargs:
            logging.info(f"Class-Balanced beta: {criterion_kwargs['beta']}")
        if 'reweight_type' in criterion_kwargs:
            logging.info(f"重加权类型: {criterion_kwargs['reweight_type']}")


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
    steps_per_epoch = len(train_loader)
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
            total_steps, cosine, final_lr, ema, use_epoch_in_loss
        )
        
        # 验证原始模型
        acc, macro_acc, report, cm = evaluate_val_set(
            model, val_loader, 
            num_classes=6, 
            class_names=class_names, 
            save_path=snapshot_path, 
            epoch=epoch
        )
        
        # 验证EMA模型
        if ema is not None:
            ema_acc, ema_macro_acc, ema_report, ema_cm = evaluate_val_set(
                ema.get_model(), val_loader, 
                num_classes=6, 
                class_names=class_names, 
                save_path=snapshot_path, 
                epoch=epoch,
                suffix="_ema"
            )
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


def setup_logger(logfile):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfgs/basicv2.yml', help='configuration file')
    parser.add_argument('--transform_cfg', type=str, default='./cfgs/transform1_224.yml', help='transform配置文件')
    parser.add_argument('--root_path', type=str, default='./data/ACDC')
    parser.add_argument('--res_path', type=str, default='./results/')
    parser.add_argument('--exp', type=str, default='ACDC/POST-NoT')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=30000)

    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--final_lr', type=float, default=1e-6, help='cosine退火的最小学习率')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup轮数')
    parser.add_argument('--cosine', action='store_true', help='使用cosine退火调度')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--test_interval_ep', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    
    parser.add_argument('--minority_boost_ratio', type=float, default=0.4, help='Boost ratio for minority class in balanced sampling')
    
    # K折交叉验证参数
    parser.add_argument('--fold', type=int, default=0, help='当前fold编号 (0-based)')
    parser.add_argument('--total_folds', type=int, default=4, help='总fold数量')
    
    # 损失函数配置参数
    parser.add_argument('--loss_type', type=str, default='label_smoothing', 
                       choices=['label_smoothing', 'rw_ldam_drw', 'ce_ldam_drw', 'ce_drw', 'focal', 'cb_focal', 'cb_ce', 'rw_ce'],
                       help='损失函数类型')
    parser.add_argument('--cls_num_list', nargs='+', type=int, default=[480, 800, 280, 30, 10, 400], 
                       help='每个类别的样本数量分布，按类别0,1,2,3...顺序排列，用于不平衡损失函数。必须根据实际数据集调整！')
    
    # LDAM相关参数
    parser.add_argument('--max_m', type=float, default=0.5, help='LDAM损失的最大边距值')
    parser.add_argument('--s', type=float, default=30, help='RW-LDAM-DRW的缩放因子')
    
    # 延迟重加权参数
    parser.add_argument('--reweight_epoch', type=int, default=80, help='开始重加权的epoch')
    parser.add_argument('--reweight_type', type=str, default='inverse', choices=['inverse', 'sqrt_inverse'], 
                       help='重加权类型')
    
    # Focal Loss参数
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal Loss的alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss的gamma参数')
    
    # Class-Balanced Loss参数
    parser.add_argument('--cb_beta', type=float, default=0.9999, help='Class-Balanced Loss的beta参数')
    
    # EMA相关参数
    parser.add_argument('--use_ema', action='store_true', default=True, help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')

    args = parser.parse_args()
    args = vars(args)
    
    cfgs_file = args['cfg']
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    for key, val in options_yaml.items():
        if isinstance(val, str) and "1e-" in val:
            options_yaml[key] = float(val)
    update_values(options_yaml, args)
    
    snapshot_path = osp.join(args["res_path"], args["exp"], f'fold_{args["fold"]}')
    os.makedirs(snapshot_path, exist_ok=True)
    
    # 创建验证结果保存的目录结构
    images_dir = osp.join(snapshot_path, 'images')
    val_txt_dir = osp.join(snapshot_path, 'val_txt')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(val_txt_dir, exist_ok=True)
    
    setup_logger(osp.join(snapshot_path, "log.txt"))
    
    if args.get('deterministic', False):
        seed = args.get('seed', 2023)
        jt.set_global_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        logging.info(f"设置随机种子: {seed}")

    # os.environ['http_proxy'] = 'http://127.0.0.1:20171'
    # os.environ['https_proxy'] = 'https://127.0.0.1:20171'


    logging.basicConfig(
        filename=osp.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    import pprint
    logging.info(pprint.pformat(args))

    run(args, snapshot_path)

