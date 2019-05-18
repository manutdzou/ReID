import torchvision.transforms as T
from .RandomErasing import RandomErasing


def transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=cfg.INPUT.BRIGHTNESS, 
              contrast=cfg.INPUT.CONTRAST, 
              saturation=cfg.INPUT.SATURATION, 
              hue=cfg.INPUT.HUE),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform