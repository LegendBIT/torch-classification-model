# 根据别人代码修改的mobilenetv2的网络模型 https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
# 权重文件下载 download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
# 此代码修改了模型结构，将第一个残差卷积模块的输出直接连接到了最后的分类层，这样做的原因是希望模型能提取更浅层的局部细节信息用于分类，
# 而不是仅仅使用深层的全局特征，从第一个残差卷积模块输出的特征经过3*3的均值池化后，拉伸为1维矢量，拼接原本的一维矢量，然后直接经过全链接进行分类
# 此外为了能够使用下载好的alpha小于1的权重文件，我们将最后一个卷积层的输出通道锁定为1280，而不是论文中的随alpha变化
from torch import nn
import torch


# 设定整个模型的所有BN层的衰减系数，该系数用于平滑统计的均值和方差，torch与tf不太一样，两者以1为互补
momentum = 0.01  # 官方默认0.1，越小，最终的统计均值和方差越接近于整体均值和方差，前提是batchsize足够大

# 保证ch可以被8整除
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# 定义基本卷积模块
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel, momentum=momentum),
            nn.ReLU6(inplace=True)
        )

# 定义mobilenetv2的基本模块
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):  # python并不要求子类一定要调用父类的构造函数
        super(InvertedResidual, self).__init__()     # 调用父类的构造函数，这里必须调用，父类的构造函数里有必须运行的代码
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel, momentum=momentum),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

# 定义模型
class MobileNetV2(nn.Module):
    def __init__(self, size_image=128, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        # last_channel2 = _make_divisible(1280 * alpha, round_nearest)
        last_channel2 = _make_divisible(1280, round_nearest)
        #################################################################################################################
        ## 残差参数1
        inverted_residual_setting1 = [
            # t, c, n, s
            [1, 16, 1, 1]
        ]
        ## 残差参数2
        inverted_residual_setting2 = [
            # t, c, n, s
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        #################################################################################################################
        ## 构建特征层1
        features1 = []
        # 定义第一层网络
        features1.append(ConvBNReLU(3, input_channel, stride=2))
        # 定义第一层残差模块
        for t, c, n, s in inverted_residual_setting1:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features1.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        output_channel1 = output_channel
        ## 构建特征层2
        features2 = []
        # 定义剩余残差模块
        for t, c, n, s in inverted_residual_setting2:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features2.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # 定义backbone的最后一层
        features2.append(ConvBNReLU(input_channel, last_channel2, 1))
        #################################################################################################################
        ## 定义backbone1
        self.features1 = nn.Sequential(*features1)
        ## 定义backbone2
        self.features2 = nn.Sequential(*features2)
        ## 定义均值池化层1
        self.avgpool1 = nn.AvgPool2d((3, 3))        
        ## 定义均值池化层2
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        ## 定义最后的分类层
        last_channel1 = int(output_channel1 * (size_image/6) * (size_image/6))
        last_channel = last_channel1 + last_channel2
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        #################################################################################################################
        # 模型权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        x1 = self.features1(input)

        x2 = self.features2(x1)
        x2 = self.avgpool2(x2)
        x2 = torch.flatten(x2, 1)     # 类比view()是改变维度，torch.flatten(input, start_dim=0, end_dim=-1) 是拉伸维度

        x1 = self.avgpool1(x1)
        x1 = torch.flatten(x1, 1)     # 类比view()是改变维度，torch.flatten(input, start_dim=0, end_dim=-1) 是拉伸维度

        x = torch.cat((x1, x2), 1)

        output = self.classifier(x)   # torch.transpose(input, dim0, dim1)是交换维度

        return output                 # 最后输出层没有激活函数，直接是全链接的输出，在损失函数中包含softmax操作，实际使用需要自己再加一个softmax
