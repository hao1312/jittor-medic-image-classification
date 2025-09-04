import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
from jimm import swin_base_patch4_window12_384
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.transforms import build_weak_strong_transforms, build_transforms
jt.flags.use_cuda = 1
jt.misc.set_global_seed(42)


# ============== Dataset ==============
class ImageFolder(Dataset):
    def __init__(self, root, annotation_path=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        if annotation_path is not None:
            with open(annotation_path, 'r') as f:
                data_dir = [line.strip().split(' ') for line in f]
            data_dir = [(x[0], int(x[1])) for x in data_dir]
        else:
            data_dir = sorted(os.listdir(root))
            data_dir = [(x, None) for x in data_dir]
        self.data_dir = data_dir
        self.total_len = len(self.data_dir)

    def __getitem__(self, idx):
        image_path, label = os.path.join(self.root, self.data_dir[idx][0]), self.data_dir[idx][1]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image_name = self.data_dir[idx][0]
        label = image_name if label is None else label
        return jt.array(image), label


def training(model: nn.Module, optimizer: nn.Optimizer, train_loader: Dataset, now_epoch: int, num_epochs: int, global_step: int,
             warmup_steps: int, base_lr: float):
    model.train()
    losses = []
    pbar = tqdm(train_loader, total=len(train_loader),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    step = 0
    for data in pbar:
        step += 1
        global_step += 1
        image, label = data

        # Warmup learning rate
        if global_step < warmup_steps:
            warmup_lr = base_lr * global_step / warmup_steps
            for group in optimizer.param_groups:
                group['lr'] = warmup_lr

        pred = model(image)
        loss = criterion(pred, label)
        loss.sync()
        optimizer.step(loss)
        losses.append(loss.item())
        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss = {losses[-1]:.4f}')

    print(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.4f}')
    return global_step


def evaluate(model:nn.Module, val_loader:Dataset):
    model.eval()
    preds, targets = [], []
    print("Evaluating...")
    for data in val_loader:
        image, label = data
        pred = model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        targets.append(label.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    acc = np.mean(np.float32(preds == targets))
    return acc


def run(model: nn.Module, train_loader: Dataset, val_loader: Dataset, num_epochs: int, modelroot: str):
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    global_step = 0
    warmup_steps = len(train_loader) * 5  # 5个epoch warmup
    base_lr = 1e-4

    for epoch in range(num_epochs):
        global_step = training(model, optimizer, train_loader, epoch, num_epochs, global_step, warmup_steps, base_lr)

        acc = evaluate(model, val_loader)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            model.save(os.path.join(modelroot, 'best.pkl'))
        if (epoch + 1) % 20 == 0:
            model.save(os.path.join(modelroot, f'epoch_{epoch + 1}.pkl'))
        model.save(os.path.join(modelroot, 'last.pkl'))

        print(f'Epoch {epoch} / {num_epochs} [VAL] best_acc = {best_acc:.2f}, acc = {acc:.2f}')


# ============== Test ==================

def test(model:nn.Module, test_loader:Dataset, result_path:str):
    model.eval()
    preds = []
    names = []
    print("Testing...")
    for data in test_loader:
        image, image_names = data
        pred = model(image)
        pred.sync()
        pred = pred.numpy().argmax(axis=1)
        preds.append(pred)
        names.extend(image_names)
    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(name + ' ' + str(pred) + '\n')

# ============== Main ==============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/root/workspace/TrainSet')
    parser.add_argument('--modelroot', type=str, default='../results/checkpoint3')
    parser.add_argument('--testonly', action='store_true', default=False)
    parser.add_argument('--loadfrom', type=str, default='../results/checkpoint3/best.pkl')
    parser.add_argument('--result_path', type=str, default='../results/checkpoint3/result.txt')
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--mean', nargs='+', type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument('--std', nargs='+', type=float, default=[0.229, 0.224, 0.225])

    args = parser.parse_args()

    data_transforms = build_transforms(args)
    transform_train = data_transforms["train"]
    transform_val = data_transforms["valid_test"]

    model = swin_base_patch4_window12_384(pretrained=True, num_classes=6)
    criterion = nn.CrossEntropyLoss()

    if not args.testonly:
        train_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/train.txt'),
            transform=transform_train,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=True
        )
        val_loader = ImageFolder(
            root=os.path.join(args.dataroot, 'images/train'),
            annotation_path=os.path.join(args.dataroot, 'labels/val.txt'),
            transform=transform_val,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False
        )
        run(model, train_loader, val_loader, 100, args.modelroot)
    else:
        test_loader = ImageFolder(
            root=args.dataroot,
            transform=transform_val,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False
        )
        model.load(args.loadfrom)
        test(model, test_loader, args.result_path)
