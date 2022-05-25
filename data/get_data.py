from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import torchvision

import configs.load_configs as config
cfg = config.load_config()

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=cfg[1]['params']['PATH_DATASETS'],
    batch_size=cfg[1]['params']['BATCH_SIZE'],
    num_workers=cfg[1]['params']['NUM_WORKERS'],
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)