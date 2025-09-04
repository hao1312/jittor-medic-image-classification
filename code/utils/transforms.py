import albumentations as A
import numpy as np
import jittor as jt
import cv2

class AlbumentationsTransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img_np = np.array(img)
        transformed = self.transform(image=img_np)
        img_np = transformed['image']
        img_np = img_np.transpose(2, 0, 1).astype('float32') / 255.0  # (C, H, W)
        return jt.array(img_np)


def build_transforms(args):
    def wrap(t): return AlbumentationsTransformWrapper(t)

    data_transforms = {
        "train": wrap(A.Compose([
            A.OneOf([
                A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=args.image_size // 20, max_width=args.image_size // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0)),

        "valid_test": wrap(A.Compose([
            A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST),
        ], p=1.0))
    }
    return data_transforms


def build_weak_strong_transforms(args):
    def wrap(t): return AlbumentationsTransformWrapper(t)

    data_transforms = {
        "train_weak": wrap(A.Compose([
            A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=args.image_size // 20, max_width=args.image_size // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0)),

        "train_strong": wrap(A.Compose([
            A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.6),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.5),
            A.CoarseDropout(max_holes=20, max_height=args.image_size // 20, max_width=args.image_size // 20,
                            min_holes=10, fill_value=0, mask_fill_value=0, p=0.7),
        ], p=1.0)),

        "valid_test": wrap(A.Compose([
            A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST),
        ], p=1.0))
    }
    return data_transforms