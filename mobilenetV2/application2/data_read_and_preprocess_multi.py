# 测试torch对于分类问题的图片及标签的读取和预处理方式
# torch的读取方式应该跟tf2.0的读取方式类似，其并不是一次将所有图片读取到内存中，而是在使用时才读取数据，原生默认的dataloader还会事先预读取一部分数据到内存
# 这部分数据的size是大于batchsize的，据说原生的dataloader会消耗完内存中的这批数据后再重新预读取一批数据，这里经过第三方的BackgroundGenerator修改后的
# dataloader则会实时维护着这批数据以提速数据读取
# 本函数的特点是增加了多类别标签的支持，其是就是读取两个标签，返回两个标签
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torchvision.transforms as transforms
import random
import os
import re
from PIL import Image
import matplotlib.pyplot as plt


# 定义自己的数据集类
class MyDataset(Dataset):
    # 初始化
    def __init__(self, path, image_size, train=True):
        if path.split(".")[-1] == "txt":
            self.images_info, self.class_names = self.read_txt(path)
        else:
            self.images_info, self.class_names = self.read_path(path)
        self.image_size = image_size
        self.train = train

        # train预处理
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),             # 以0.5的概率随机水平翻转图片，输入必须是PIL Image
            # transforms.RandomVerticalFlip(),             # 以0.5的概率随机垂直翻转图片，输入必须是PIL Image
            transforms.GaussianBlur(5, sigma=(0.1, 0.5)),  # 高斯模糊, 模糊半径越大, 正态分布标准差越大, 图像就越模糊, 方差介于sigma之间
            # transforms.RandomAffine(2),                    # 随机旋转(平移+缩放+扭曲)
            transforms.ColorJitter(brightness=(1, 1.2), contrast=(1, 1.2), saturation=(1, 1.2), hue=0.05),  # 随机改变图片的亮度，对比度和饱和度
            #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # 图片锐化，当前版本torch不能用
            transforms.ToTensor(),                         # 接受PIL Image或numpy.ndarray格式, 先由HWC转置为CHW格式, 再转为float类型, 最后，每个像素除以255
            transforms.Normalize(mean=[0.5], std=[0.5])    # 减均值除方差，在做数据归一化之前必须要把PIL Image转成Tensor, 而其他resize或crop操作则不需要
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # imagenet
        ])

        # test预处理
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # imagenet
        ])

    # 基于index执行单张图片的读取和预处理
    def __getitem__(self, index):
            img_path, label1, label2 = self.images_info[index]
            image = Image.open(img_path)
            image = self.resize_padding(image, self.image_size)
            label1, label2 = int(label1), int(label2)

            # 注意区分预处理
            if self.train:
                img = self.train_transforms(image)
            else:
                img = self.test_transforms(image)

            if label2 > 9:
                label2 = int(10)

            return img, label1, label2

    # 返回数据集长度
    def __len__(self):
            return len(self.images_info)

    # 等比例拉伸图片，多余部分填充value
    def resize_padding(self, image, target_length, value=0):
        h, w = image.size                                   # 获得原始尺寸
        ih, iw = target_length, target_length               # 获得目标尺寸
        scale = min(iw/w, ih/h)                             # 实际拉伸比例
        nw, nh  = int(scale * w), int(scale * h)            # 实际拉伸后的尺寸
        image_resized = image.resize((nh, nw), Image.ANTIALIAS)    # 实际拉伸图片
        image_paded = Image.new("RGB", (ih, iw), value)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded.paste(image_resized, (dh, dw, nh+dh, nw+dw))   # 居中填充图片
        return image_paded

    # 读取txt文档，文档结构如下
    # car dog horse
    # /123/456/7891.jpg 0
    # /123/456/7892.jpg 1
    # /123/456/7893.jpg 2
    def read_txt(self, txt_path):
        print("--> Loading file paths and labels...")
        with open(txt_path, 'r', encoding='utf-8') as f:
            dataset_info = f.readlines()
        images_info_list = list(map(lambda x:re.split(r"[ ]+", x.strip()), dataset_info[1:]))
        random.seed(random.randint(0, 666))           # 为了将读取的图片列表和标签列表打乱
        random.shuffle(images_info_list)
        print("<-- Loading end...（total num = %d）" % len(images_info_list))
        return images_info_list, dataset_info[0].strip().split(' ')
    
    # 得到dir路径下所有文件的具体路径及标签，file_paths_list, labels_list都是list分别对应着每个文件的具体路径和标签编号，class_names对应标签名  
    def read_path(self, dir):
        print("--> Loading file paths and labels...")
        image_format = ["bmp", "jpg", "jpeg", "png"]  # 可以读取的文件类型
        file_paths_list, labels_list, class_names = [], [], []
        for path, _, files in os.walk(dir):           # path是dir路径下每个文件夹的路径和dir文件夹本身的路径，files是每个文件夹下的文件名列表
            if path == dir: continue                  # dir本身路径不要
            for file in files:
                if file.split(".")[-1] not in image_format:           # 在mac中每个文件夹里有一个隐藏的.DS_Store，予以删除
                    continue
                file_paths_list.append(os.path.join(path, file))
                labels_list.append(path.split("/")[-1])
            class_names.append(path.split("/")[-1])
        class_names.sort()                            # 标签名按照字母顺序排列
        labels_list = [class_names.index(label) for label in labels_list]
        randnum = random.randint(0, 666)              # 为了将读取的图片列表和标签列表打乱
        random.seed(randnum)
        random.shuffle(file_paths_list)
        random.seed(randnum)
        random.shuffle(labels_list)
        print("<-- Loading end...（total num = %d）" % len(labels_list))
        return list(zip(file_paths_list, labels_list)), class_names


# 定义自己的数据集读取类, 原本PyTorch默认的DataLoader会创建一些worker线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据
# 使用prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载
class MyDataLoader(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == "__main__":

    dir = "./test_multi.txt"
    my_dataset = MyDataset(dir, 128)
    my_loader = MyDataLoader(dataset=my_dataset, batch_size=16, shuffle=True, num_workers=4)

    plt.figure(figsize=(10, 10))
    for images, labels1, labels2 in my_loader:
        print(images.size())
        print(type(images))
        print(labels1)
        print(labels2)
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(((images[i].numpy().transpose(1,2,0)*0.5+0.5)*255).astype("uint8"))
            plt.title(my_dataset.class_names[labels1[i]] + " + " + my_dataset.class_names[labels2[i]])
            plt.axis("off")
        break
    plt.show()
    plt.savefig('./test.png', format='png')
