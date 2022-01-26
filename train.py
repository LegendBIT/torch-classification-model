# 可实现迁移学习
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ghost import ghostnet
from data_read_and_preprocess import MyDataset, MyDataLoader
import os
import shutil
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
############################################################################################################
## 0. 参数设置 ##############################################################################################
############################################################################################################
IMG_SIZE = 128
BATCH_SIZE = 32
CLASS_NUM = 2
alpha = 1.0           # 模型通道缩放系数
initial_epochs = 4    # 第一轮仅仅训练最后一层
second_epochs = 6     # 第二轮训练整个网络
initial_learning_rate = 0.0001
second_learning_rate = 0.00001
dataset_name = "Data"
# train_dir = "./train"
# test_dir = "./test"
# output_path = "./checkpoint/pth/"
weight_path = "./checkpoint/state_dict_73.98.pth"

train_dir = "./train"
test_dir = "./validation"
output_path = "./"

############################################################################################################
## 1. 读取数据和数据预处理 #####################################################################################
############################################################################################################
train_dataset = MyDataset(train_dir, IMG_SIZE)
train_loader  = MyDataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataset = MyDataset(test_dir, IMG_SIZE, train=False)
test_loader  = MyDataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

############################################################################################################
## 2. 搭建网络结构 ###########################################################################################
############################################################################################################
model = ghostnet(num_classes=CLASS_NUM, width=alpha)
model.to(device)
#print(model)

############################################################################################################
## 3. 定义损失函数，日志记录器和模型保存函数 ######################################################################
############################################################################################################
cost = torch.nn.CrossEntropyLoss()
tensorboard_path = "./log"
if os.path.exists(tensorboard_path): shutil.rmtree(tensorboard_path)
writer = SummaryWriter(tensorboard_path)
# 保存模型参数和结构
def save_model(epoch, acc1, acc2, path, dataset_name, model, alpha):
    localtime = time.strftime("%Y%m%d-%H%M", time.localtime())
    output_model = path + "{}_ghost-v0.1_{}_{}_{}_{}_{:.4f}_{:.4f}.pth".format(dataset_name, localtime, IMG_SIZE, alpha, epoch, acc1, acc2)
    torch.save(model, output_model)

############################################################################################################
## 4. 定义训练函数和测试函数 ###################################################################################
############################################################################################################
def train(model, train_loader, optimizer, epoch):
    sum_loss = 0.0      # 统计每个epoch的总loss
    train_correct = 0   # 统计每个epoch的correct
    num = 0
    for inputs, labels in train_loader:
        if str(device).split(":")[0]=="cuda": inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()          # 清空梯度
        outputs = model(inputs)        # 计算输出，并缓存中间每层输出
        loss = cost(outputs, labels)   # 计算损失
        loss.backward()                # 损失梯度反向传输，损失梯度结合缓存的中间层输出计算出所有权重参数的梯度
        optimizer.step()               # 结合梯度更新参数

        _, id = torch.max(outputs.data, 1)  # 完整tensor包含数据data,梯度grad,求导记录grad_fn，tensor.data指的就是数据，max函数可以用于完整tensor,也可用于数据
        sum_loss += loss.data               # tensor.item()转换1*1的张量的数据data为fp数值，tensor.cpu().numpy()转换data为ndarray
        train_correct += torch.sum(id == labels.data)/id.size()[0]  # tensor除了以上数值，还包含一个requires_grad属性

        num += 1                       # 上下几行代码中.data和.item()都可以去掉而没有影响，.data返回还是tensor, .item()返回是fp数值
        writer.add_scalar('Train/Lr', optimizer.param_groups[0]['lr'], len(train_loader)*epoch+num)  # 输入数值或者tensor都可以
        writer.add_scalar('Train/Loss', loss.item(), len(train_loader)*epoch+num)      # tensor.item()会自动将数据从GPU移动到CPU
        writer.add_scalar('Train/Accuracy', torch.sum(id == labels.data).cpu()/id.cpu().numpy().size, len(train_loader)*epoch+num)
        writer.flush()
    
    return sum_loss, train_correct/len(train_loader)

def test(model, test_loader):
    model.eval()  # 测试模式，仅用于通知BN层和dropout层当前处于推理模式还是训练模式，不影响梯度计算和反向传播
    with torch.no_grad():  # 放弃梯度记录节省内存，等效于设置requires_grad=False，正向传播时不记录中间层结果，无法进行梯度计算和更新参数
        sum_loss = 0.0
        test_correct = 0
        for inputs, labels in test_loader:
            if str(device).split(":")[0]=="cuda": inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = cost(outputs, labels)
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            test_correct += torch.sum(id == labels.data)/id.size()[0]
    return sum_loss, test_correct/len(test_loader)

############################################################################################################
## 5. 训练神经网络 ###########################################################################################
############################################################################################################
# 定义函数单独设置所有BN层为推理模式
def freeze_bn(m): 
    if isinstance(m, nn.BatchNorm2d): m.eval() 
# model.apply(freeze_bn)

# 读取权重文件，读取为有序字典的形式
assert os.path.exists(weight_path), "file {} dose not exist.".format(weight_path)
pre_weights = torch.load(weight_path, map_location=device)

# 如果最后一层的类别数量相同，则恢复整个模型的权重，如果不相等，则只是恢复前n-1层的权重
pre_dict = {k: v for k, v in pre_weights.items() if model.state_dict()[k].numel() == v.numel()}
missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

# 冻结n-2层，方式就是设置层的requires_grad为False, model.named_parameters()是列表不包含BN的滑动均值和方差，仅仅包含可训练的参数
freeze_list=list(model.state_dict().keys())[0:-2]   # model.state_dict()是有序字典，包含所有参数
for name, param in model.named_parameters():        # print(model.state_dict()['features.0.1.running_mean'][0:5])
    if name in freeze_list: param.requires_grad=False

# 预先计算loss和acc
loss0, acc0 = test(model, test_loader)
print("initial loss: {:.4f}".format(loss0))
print("initial accuracy: {:.4f}".format(acc0))

# 选择需要训练的参数，其实只要设置了requires_grad属性为false，即使选择所有参数可训练，也不会更新参数了，但是下面写法更科学
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=initial_learning_rate)
# optimizer.param_groups[0]['lr'] = initial_learning_rate   # 更改学习率

# 第一轮训练
for epoch in range(initial_epochs) :
    model.train()
    model.apply(freeze_bn)   # 因为是迁移学习，所以选择冻结BN层的滑动均值和滑动方差
    loss, acc = train(model, train_loader, optimizer, epoch)
    val_loss, val_acc = test(model, test_loader)
    save_model(epoch+1, acc, val_acc, "./tmp/", dataset_name, model, alpha)
    print("epoch: %d, loss: %0.5f, acc: %0.4f, val_loss: %0.5f, val_acc: %0.4f" % (epoch+1, loss, acc, val_loss, val_acc))
    print("")

# 计算loss和acc
loss1, acc1 = test(model, test_loader)
print("middle loss: {:.4f}".format(loss1))
print("middle accuracy: {:.4f}".format(acc1))

# 解冻所有层
for name, param in model.named_parameters(): param.requires_grad=True
optimizer = torch.optim.Adam(model.parameters(), lr=second_learning_rate)

# 开始第二轮训练
for epoch in range(initial_epochs, initial_epochs+second_epochs) :
    model.train()
    model.apply(freeze_bn)
    loss, acc = train(model, train_loader, optimizer, epoch)
    val_loss, val_acc = test(model, test_loader)
    save_model(epoch+1, acc, val_acc, "./tmp/", dataset_name, model, alpha)
    print("epoch: %d, loss: %0.5f, acc: %0.4f, val_loss: %0.5f, val_acc: %0.4f" % (epoch+1, loss, acc, val_loss, val_acc))
    print("")

# 计算loss和acc
loss2, acc2 = test(model, test_loader)
print("last loss: {:.4f}".format(loss2))
print("last accuracy: {:.4f}".format(acc2))

# 保存模型参数和结构
save_model(initial_epochs+second_epochs, acc, acc2, output_path, dataset_name, model, alpha)


# 6. 遍历模型
# model.cpu()
# parm = {}
# print("Model's named_parameters:")              # model.named_parameters()是列表[(name, tensor), ...]仅仅包含可训练的参数
# for name, parameters in model.named_parameters():
#     print(name, "\t", parameters.size())
#     parm[name] = parameters.detach().numpy()    # .detach等效于.data, 但是detach更安全建议使用这个

# print("Model's state_dict:")                    # model.state_dict()是字典{name:tesnor, ...}包含所有参数，包括BN的滑动均值和方差
# for param_tensor in model.state_dict():
#     print(param_tensor,"\t", model.state_dict()[param_tensor].size())

# print(type(list(model.named_parameters())[0]))   # ('features.0.0.weight', <class 'torch.nn.parameter.Parameter'>)
# print(type(list(model.parameters())[0]))         # <class 'torch.nn.parameter.Parameter'>
# print(model.state_dict()['features.0.0.weight']) # <class 'torch.nn.parameter.Parameter'>

# print(type(model.named_parameters()))            # <class 'generator'>
# print(type(model.parameters()))                  # <class 'generator'>
# print(type(model.state_dict()))                  # <class 'collections.OrderedDict'> 
