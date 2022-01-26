# torch模型的数据集测试
import time
import logging
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from data_read_and_preprocess import MyDataset, MyDataLoader

logging.basicConfig(level=logging.NOTSET)


def main():

    # 初始化
    IMG_SIZE = 128
    BATCH_SIZE = 32
    num_gpu = 1
    test_dir = "./cats_and_dogs_filtered/validation"
    weight_path = "./x.pth"

    # 读取模型
    model = torch.load(weight_path)
    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu))).cuda()
    print(">>> Model created.")

    # 创建数据集
    test_dataset = MyDataset(test_dir, IMG_SIZE, train=False)
    test_loader  = MyDataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(">>> Data_loader created.")

    # 创建损失函数
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    print(">>> Loss_fn created.")

    # 测试模型
    eval_metrics = validate(model, test_loader, validate_loss_fn)
    print(eval_metrics)

# 测试函数
def validate(model, loader, loss_fn, log_suffix=''):
    # 需要统计的参量
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    prec_recall = PrecAndRecall(loader.dataset.class_names)
    # 设定为推理模式
    model.eval()
    # 循环处理每一批数据
    end = time.time()           # 在链式求导法则中，某层的某权重的梯度等于该层该权重对应的误差梯度乘以该层该权重对应的输入，
    last_idx = len(loader) - 1  # 所以如果想要求梯度，那在正向推理时需要记录所有的中间结果，即中间每一层的输出都要记录下来，
    with torch.no_grad():  # 如果后面无需求梯度，那可以通过设定torch.no_grad()，使得前向推理时不记录中间每一层的输出结果，
        for batch_idx, (input, target) in enumerate(loader):                              # 以节省内存加快速度
            last_batch = batch_idx == last_idx     # 数据总长度不一定是batchsize的整数倍，所以记录是否最后一个batch
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 2))

            losses_m.update(loss.item(), input.size(0))     # tensor.item()会自动将数据从GPU移动到CPU
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            prec_recall.update(output, target, output.size(0))

            torch.cuda.synchronize()                  # torch是异步调用机制，所以需要同步统计pytorch调用cuda运行时间
            batch_time_m.update(time.time() - end)    # 因为CUDA kernel函数是异步的，所以不能直接在CUDA函数两端加上time.time()测试时间，
            end = time.time()                         # 这样测出来的只是调用CUDA api的时间，不包括GPU端运行的时间。

            if (last_batch or batch_idx % 20 == 0):
                log_name = 'Test' + log_suffix
                logging.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    prec_recall.result()

    return metrics

# 计算topk的准确度, output - batchsize*nclass, target - nclass*1 or nclass, topk - n
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)                               # size 1
    batch_size = target.size(0)                    # size 1
    _, pred = output.topk(maxk, 1, True, True)     # size batchsize*maxk
    pred = pred.t()                                # size maxk*batchsize
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # size maxk*batchsize, 每一列代表一个输入对应的输出结果
    return [correct[:k].contiguous().view(-1).float().sum(0) * 100. / batch_size for k in topk]  # 这一列中的每一行对应着topk
    # return 长度为n的list, 对应着topk=(1,), 分别表示对应的topk的准确度

# 结果记录类，存储当前值，平均值，总和，总共处理样本数目
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 计算精确率和召回率
class PrecAndRecall(object):
    def __init__(self, label_name):
        self.label_name = label_name
        self.num_prs = [0.000000001 for _ in range(len(label_name))]
        self.num_rec = [0.000000001 for _ in range(len(label_name))]
        self.num_mat = [[0] * len(label_name) for _ in range(len(label_name))]
        self.label_true, self.label_pred = [], []
    
    def update(self, output, labels, n):
        _, pred = output.topk(1, 1, True, True)     # size batchsize*1
        pred = pred.cpu()
        labels = labels.cpu()
        for i in range(n):
            self.num_prs[int(pred[i])] += 1
            self.num_rec[int(labels[i])] += 1
            self.num_mat[int(labels[i])][int(pred[i])] += 1
            self.label_true.append(int(labels[i]))
            self.label_pred.append(int(pred[i]))

    def result(self):
        # 计算总的准确率accuracy
        num1, num2, num3 = 0, 0, 0
        for i in range(len(self.label_name)):
            num1 += self.num_mat[i][i]
            num2 += self.num_prs[i]
            num3 += self.num_rec[i]
        assert int(num2) == int(num3)
        accuracy = (num1/num2*100*100//1)/100
        print("total accuracy: "+ str(accuracy))

        # 计算每个类别的precision
        result_prs = {}
        prs = [float(self.num_mat[i][i])/float(self.num_prs[i])*100 for i in range(len(self.label_name))]
        for i in range(len(self.label_name)):
            result_prs[self.label_name[i]] = (prs[i]*100//1)/100
        print("every precision: "+ str(result_prs))

        # 计算每个类别的recall
        result_rec = {}
        rec = [float(self.num_mat[i][i])/float(self.num_rec[i])*100 for i in range(len(self.label_name))]
        for i in range(len(self.label_name)):
            result_rec[self.label_name[i]] = (rec[i]*100//1)/100
        print("every recall:   "+ str(result_rec))

        # 打印混淆矩阵
        print(self.label_name)
        print(np.array(self.num_mat))



if __name__ == '__main__':
    main()
