"""CNN architectures for four-season colour classification.

"""

import torch.nn as nn
import torchvision


class SeasonCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResNetTransfer(nn.Module):
    """Transfer-learning wrapper around a torchvision ResNet18 pretrained on
    ImageNet, with its final fc layer replaced for num_classes.

    Opt-in via train.py's --arch resnet18 (default stays "seasoncnn" above)
    so this can be disabled just by not passing that flag.

    freeze_backbone=True (the default) trains only the new fc layer -- fast
    and less prone to overfitting on ~4k training images. Pass
    freeze_backbone=False (train.py's --finetune-backbone) to unfreeze every
    layer for a slower, more thorough fine-tune.
    """

    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.backbone = torchvision.models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
