# torch预测单张图片
import torch
import torchvision.transforms as transforms
from PIL import Image


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
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

# 读取图片并预处理
image = resize_padding(Image.open("/nfs/cache-902-2/xiawenze/tmp/mobilenetv2_torch/images/car_1.jpg"), 128)
image = transform(image)
image = image.reshape(1, 3, 128, 128)

# 读取模型
model = torch.load("/nfs/cache-902-2/xiawenze/tmp/mobilenetv2_torch/tmp/D2city_BDD_ZJC_mobilenetv2-v0.1_20220120-2156_128_1.0_9_0.8886_0.8856.pth")
# print(model)

# 单张图片推理
model.cpu().eval()
with torch.no_grad():
    output = torch.squeeze(model(image))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

# 输出结果
label = ["background", "bus", "car", "person", "rider", "truck", "two-wheeled"]
print(predict[predict_cla])
print(label[predict_cla])
