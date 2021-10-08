import torch
import torch.nn as nn

from .module import InvertedResidual, MobileVitBlock

model_cfg = {
    "xxs":{
        "features": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        "d": [64, 80, 96],
        "expansion_ratio": 2,
        "layers": [2, 4, 3]
    },
    "xs":{
        "features": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "d": [96, 120, 144],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
    "s":{
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
}

class MobileViT(nn.Module):
    def __init__(self, img_size, features_list, d_list, transformer_depth, expansion, num_classes = 1000):
        super(MobileViT, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = features_list[0], kernel_size = 3, stride = 2, padding = 1),
            InvertedResidual(in_channels = features_list[0], out_channels = features_list[1], stride = 1, expand_ratio = expansion),
        )

        self.stage1 = nn.Sequential(
            InvertedResidual(in_channels = features_list[1], out_channels = features_list[2], stride = 2, expand_ratio = expansion),
            InvertedResidual(in_channels = features_list[2], out_channels = features_list[2], stride = 1, expand_ratio = expansion),
            InvertedResidual(in_channels = features_list[2], out_channels = features_list[3], stride = 1, expand_ratio = expansion)
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(in_channels = features_list[3], out_channels = features_list[4], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[4], out_channels = features_list[5], d_model = d_list[0],
                           layers = transformer_depth[0], mlp_dim = d_list[0] * 2)
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(in_channels = features_list[5], out_channels = features_list[6], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[6], out_channels = features_list[7], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 4)
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(in_channels = features_list[7], out_channels = features_list[8], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[8], out_channels = features_list[9], d_model = d_list[2],
                           layers = transformer_depth[2], mlp_dim = d_list[2] * 4),
            nn.Conv2d(in_channels = features_list[9], out_channels = features_list[10], kernel_size = 1, stride = 1, padding = 0)
        )

        self.avgpool = nn.AvgPool2d(kernel_size = img_size // 32)
        self.fc = nn.Linear(features_list[10], num_classes)


    def forward(self, x):
        # Stem
        x = self.stem(x)
        # Body
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # Head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def MobileViT_XXS(img_size = 256, num_classes = 1000):
    cfg_xxs = model_cfg["xxs"]
    model_xxs = MobileViT(img_size, cfg_xxs["features"], cfg_xxs["d"], cfg_xxs["layers"], cfg_xxs["expansion_ratio"], num_classes)
    return model_xxs

def MobileViT_XS(img_size = 256, num_classes = 1000):
    cfg_xs = model_cfg["xs"]
    model_xs = MobileViT(img_size, cfg_xs["features"], cfg_xs["d"], cfg_xs["layers"], cfg_xs["expansion_ratio"], num_classes)
    return model_xs

def MobileViT_S(img_size = 256, num_classes = 1000):
    cfg_s = model_cfg["s"]
    model_s = MobileViT(img_size, cfg_s["features"], cfg_s["d"], cfg_s["layers"], cfg_s["expansion_ratio"], num_classes)
    return model_s


if __name__ == "__main__":
    img = torch.randn(1, 3, 256, 256)

    cfg_xxs = model_cfg["xxs"]
    model_xxs = MobileViT(256, cfg_xxs["features"], cfg_xxs["d"], cfg_xxs["layers"], cfg_xxs["expansion_ratio"])

    cfg_xs = model_cfg["xs"]
    model_xs = MobileViT(256, cfg_xs["features"], cfg_xs["d"], cfg_xs["layers"], cfg_xs["expansion_ratio"])

    cfg_s = model_cfg["s"]
    model_s = MobileViT(256, cfg_s["features"], cfg_s["d"], cfg_s["layers"], cfg_s["expansion_ratio"])

    print(model_s)

    # XXS: 1.3M 、 XS: 2.3M 、 S: 5.6M
    print("XXS params: ", sum(p.numel() for p in model_xxs.parameters()))
    print(" XS params: ", sum(p.numel() for p in model_xs.parameters()))
    print("  S params: ", sum(p.numel() for p in model_s.parameters()))