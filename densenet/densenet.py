# 根据torch官方代码修改的densenet代码
# 模型下载地址：
#           121 --- https://download.pytorch.org/models/densenet121-a639ec97.pth
#           161 --- https://download.pytorch.org/models/densenet161-8d451a50.pth
#           169 --- https://download.pytorch.org/models/densenet169-b2777c0a.pth
#           201 --- https://download.pytorch.org/models/densenet201-c1103571.pth

from collections import OrderedDict
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 定义了当你使用 from <module> import * 导入某个模块的时候能导出的符号
__all__ = [
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]

# 定义densenet的最基本模块，包含BN1 + relu1 + conv1 + BN2 + relu2 + conv2 + dropout，注意这里是BN在最前面，一般别的模型都是conv在前
class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1: nn.BatchNorm2d                                    # 定义norm1这个字段并提前赋予数据类型
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))  # 对定义的norm1字段进行赋值
        self.relu1: nn.ReLU
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module(                                              # 第一个卷积模块输出通道数是bn_size * growth_rate
            "conv1", nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.norm2: nn.BatchNorm2d
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module(                                              # 第二个卷积模块输出通道数是growth_rate
            "conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.drop_rate = float(drop_rate)

    def forward(self, input: Tensor) -> Tensor:
        prev_features = input
        new_features0 = torch.cat(prev_features, 1)                   # 每一个最基本模块的输入通道是init_num + (n - 1) * growth_rate
        new_features1 = self.conv1(self.relu1(self.norm1(new_features0)))  # 第一个卷积输出通道bn_size * growth_rate
        new_features2 = self.conv2(self.relu2(self.norm2(new_features1)))  # 第二个卷积输出通道growth_rate
        if self.drop_rate > 0:                                        # 每一个最基本模块的输出通道是growth_rate
            new_features2 = F.dropout(new_features2, p=self.drop_rate, training=self.training)  # 当前使用时，没有启用这一层
        return new_features2

# 定义densenet的大模块，包含num_layers个最基本模块，这个num_layers个最基本模块遵循密集连接的原则
# nn.ModuleDict可以以字典的形式向nn.ModuleDict中输入子模块，也可以以add_module()的形式向nn.ModuleDict中输入子模块
# 但是nn.ModuleDict类似于nn.Module需要自己实现forward()函数，类似的模块还有nn.ModuleList以列表形式搭建模型
# 所以说白了nn.Sequential，nn.Module，nn.ModuleList，nn.ModuleDict是搭建模型或模块的四种方式，是并行的关系，可以根据不同应用条件下使用
# https://blog.csdn.net/weixin_42486623/article/details/122822580
class _DenseBlock(nn.ModuleDict):

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)   # 以add_module()形式输入子模块

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():                       # 以items()形式访问子模块
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

# 定义densenet的大模块，用于拼接_DenseBlock模块，在本模块内通过均值池化将空间尺寸减小一半
# torch.nn.Sequential相当于tf2.0中的keras.Sequential()，其实就是以最简单的方式搭建序列模型，不需要写forward()函数，
# 直接以列表形式将每个子模块送进来就可以了，或者也可以使用OrderedDict()或add_module()的形式向模块中添加子模块
# https://blog.csdn.net/weixin_42486623/article/details/122822580
class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))

# 根据block_config参数配置列表搭建整个densenet模型
class DenseNet(nn.Module):

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ) -> None:

        super().__init__()

        ### 搭建第一层，即stem层，包含conv + BN + relu + maxpool，以字典的形式向nn.Sequential中添加子模块
        self.features = nn.Sequential(        # 用nn.Sequential搭建一个子模块，不需要重写forward()函数
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        ### 搭建bottleneck层，包含4个_DenseBlock大模块和4个_Transition大模块
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2  # _Transition模块不仅将空间尺寸减半还将通道尺寸减半
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        ### 搭建最后的分类层
        self.classifier = nn.Linear(num_features, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

##############################################################################################################################
## 通过修改配置列表实现不同模型的定义
def densenet121(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 24, 16), 64, **kwargs)

def densenet161(**kwargs: Any) -> DenseNet:
    return DenseNet(48, (6, 12, 36, 24), 96, **kwargs)

def densenet169(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 32, 32), 64, **kwargs)

def densenet201(**kwargs: Any) -> DenseNet:
    return DenseNet(32, (6, 12, 48, 32), 64, **kwargs)

if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image
    import re

    # 等比例拉伸图片，多余部分填充value
    def resize_padding(image, target_length, value=0):
        h, w = image.size                                   # 获得原始尺寸
        ih, iw = target_length, target_length               # 获得目标尺寸
        scale = min(iw/w, ih/h)                             # 实际拉伸比例
        nw, nh  = int(scale * w), int(scale * h)            # 实际拉伸后的尺寸
        image_resized = image.resize((nh, nw), Image.ANTIALIAS)    # 实际拉伸图片
        image_paded = Image.new("RGB", (ih, iw), value)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded.paste(image_resized, (dh, dw, nh+dh, nw+dw))   # 居中填充图片
        return image_paded

    # 变换函数
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 读取图片并预处理
    image = resize_padding(Image.open("./car.jpg"), 224)
    image = transform(image)
    image = image.reshape(1, 3, 224, 224)

    # 建立模型并恢复权重
    weight_path = "./checkpoint/densenet121-a639ec97.pth"  # 这个预训练权重是老版本torch生成的，当时模块的命名允许出现"."
    pre_weights = torch.load(weight_path)  # 但是最新的torch不允许出现"."，所以老版权重恢复进新版模型时需要修改一下模块命名
    pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
    for key in list(pre_weights.keys()):   # 主要是新版模型中的最基础模块的命名是类似于...denselayer1.conv1.weight
        res = pattern.match(key)           # 而老版本权重的命名类似于               ...denselayer1.conv.1.weight
        if res:                            # 所以需要正则表达式去老版本权重的key中匹配一下，一旦匹配成功就修改为最新模型的权重名称
            new_key = res.group(1) + res.group(2)  # 正则表达式中()的作用是提取满足匹配要求的字符串，group(0)就是匹配正则表达式整体结果
            pre_weights[new_key] = pre_weights[key]
            del pre_weights[key]
    model = densenet121()
    model.load_state_dict(pre_weights)
    # print(model)

    # 单张图片推理
    model.cpu().eval()     # .eval()用于通知BN层和dropout层，采用推理模式而不是训练模式
    with torch.no_grad():  # torch.no_grad()用于整体修改模型中每一层的requires_grad属性，使得所有可训练参数不能修改，且正向计算时不保存中间过程，以节省内存
        output = torch.squeeze(model(image))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 输出结果
    print(predict_cla)
    print(predict[predict_cla])
