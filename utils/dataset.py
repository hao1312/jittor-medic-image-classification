import os
from PIL import Image
import jittor as jt

from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, ImageNormalize
from sklearn.model_selection import KFold
import logging

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

def load_train_val_samples(root_path):
    def read_samples(txt_path):
        samples = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # 支持逗号或空格分隔
                if ',' in line:
                    img, label = line.split(',')
                else:
                    img, label = line.split()
                samples.append((img.strip(), int(label.strip())))
        return samples

    train_txt = os.path.join(root_path, 'labels/train.txt')
    val_txt = os.path.join(root_path, 'labels/val.txt')
    train_samples = read_samples(train_txt)
    val_samples = read_samples(val_txt)
    return train_samples, val_samples


def split_kfold_dataset(annotation_path, fold=0, num_folds=5, shuffle=True, seed=42):
    """
    annotation_path: Path to txt file, each line: 'filename label'
    fold: which fold to return (0~fold_num-1)
    Returns: train_list, val_list (list of (filename, label))
    """
    with open(annotation_path, 'r') as f:
        all_data = [line.strip().split(' ') for line in f]
    all_data = [(x[0], int(x[1])) for x in all_data]

    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
    splits = list(kf.split(all_data))
    train_idx, val_idx = splits[fold]
    train_list = [all_data[i] for i in train_idx]
    val_list = [all_data[i] for i in val_idx]
    # print(f"train_list: {train_list}, val_list: {val_list}")
    return train_list, val_list


class ImageFolder2(Dataset):
    def __init__(self, root, samples=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.transform = transform
        self.samples = samples
        self.total_len = len(self.samples)
        
        # 设置Jittor框架必需的属性
        self.real_len = self.total_len
        self.real_batch_size = self.batch_size if hasattr(self, 'batch_size') else 16
        self.batch_len = (self.total_len + self.real_batch_size - 1) // self.real_batch_size
        
        # 设置数据集属性
        self.set_attrs(total_len=self.total_len)

    def __getitem__(self, idx):
        image_path, label = os.path.join(self.root, self.samples[idx][0]), self.samples[idx][1]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.samples[idx][0] if label is None else label
        return jt.array(image), label

