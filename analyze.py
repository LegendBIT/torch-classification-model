# 循环处理测试集中的每一张图片，按照正确或错误识别结论，分别存储每一张图片，并分析不同置信度段的正确与错误识别数量
import torch
import torchvision.transforms as transforms
from PIL import Image
import re
import os
import shutil
import json


# 获取txt中的图像数据
def get_images(txt_path):
    print("--> Loading file paths and labels...")
    with open(txt_path, 'r', encoding='utf-8') as f:
        dataset_info = f.readlines()
    images_info_list = list(map(lambda x:re.split(r"[ ]+", x.strip()), dataset_info[1:]))
    images_labels = dataset_info[0].strip().split(' ')
    print("<-- Loading end...（total num = %d）" % len(images_info_list))
    return images_info_list, images_labels

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

# 处理单张图片
def proc_images(image_path, transform):
    # image = resize_padding(Image.open(image_path), 128)
    image = Image.open(image_path)
    image = transform(image)
    image = image.reshape(1, 3, 48, 48)
    return image

# 复制图片
def copy_image(path1, path2):
    f1 = open(path1, 'rb')       # rb 以二进制读取
    data = f1.read()             # 先把图片读出来
    f2 = open(path2, 'wb')       # 创建一个新的文件
    f2.write(data)               # 写入图片
    f1.close()
    f2.close()

# 分析和存储每一张图片的识别结果
def analyze_result(result_analysis_sim_cls, result_analysis_mid_cls, result_analysis_dif_cls, images_labels, path_tmp, 
                                            image_path, label_id, pred_id, pred_prob, save_images_true, save_images_false): 
   ## ———————————————————————————————————— 统计检测网络和分类网络的得分分析 ————————————————————————————————————————— ##
    r_cls = min(int(pred_prob*10), 9)
    c_cls = int(pred_id==label_id)
    result_analysis_sim_cls[r_cls][c_cls] += 1
    result_analysis_mid_cls[r_cls][label_id][c_cls] += 1
    result_analysis_dif_cls[r_cls][label_id][pred_id] += 1       # 直接是混淆矩阵

    ## ————————————————————————————————————————— 保存所有中间结果图片 —————————————————————————————————————————————— ##
    if save_images_false and c_cls == 0:          # 保存分类的错误结果
        if result_analysis_mid_cls[r_cls][label_id][c_cls] == 1:
            tmp = path_tmp + "false/" + str(r_cls) + "/" + images_labels[label_id] + "/"
            if os.path.exists(tmp):
                shutil.rmtree(tmp)                # 删除现有路径，os.rmdir(tmp)函数只能用于删除空目录
            os.makedirs(tmp)                      # 新建立路径
        image_path_cls = path_tmp + "false/" + str(r_cls) + "/" + images_labels[label_id] + "/" \
                        + images_labels[pred_id] + "_" + str(((pred_prob.tolist()*100)//1)/100) + "_" + image_path.split("/")[-1]
        copy_image(image_path, image_path_cls)
    if save_images_true and c_cls == 1:           # 保存分类的正确结果
        if result_analysis_mid_cls[r_cls][label_id][c_cls] == 1:
            tmp = path_tmp + "true/" + str(r_cls) + "/" + images_labels[label_id] + "/"
            if os.path.exists(tmp):
                shutil.rmtree(tmp)                # 删除现有路径，os.rmdir(tmp)函数只能用于删除空目录
            os.makedirs(tmp)                      # 新建立路径
        image_path_cls = path_tmp + "true/" + str(r_cls) + "/" + images_labels[label_id] + "/" \
                        + images_labels[pred_id] + "_" + str(((pred_prob.tolist()*100)//1)/100) + "_" + image_path.split("/")[-1]
        copy_image(image_path, image_path_cls)

## ————————————————————————————————————————— 打印中间得分分析结果 —————————————————————————————————————————————— ## 
def print_result(result_analysis_sim_cls, result_analysis_mid_cls, result_analysis_dif_cls, images_labels): 
    print("==="*30)
    print("classification simple analysis")
    print("==="*30)
    for i in range(10):
        print("classification ({:.1f} ~ {:.1f}): {:<12s}".format(i/10, (i+1)/10, str(result_analysis_sim_cls[i])))

    print("==="*30)
    print("classification middle analysis")
    print("==="*30)
    for i in range(10):
        print("classification ({:.1f} ~ {:.1f}): {}".format(i/10, (i+1)/10, str(result_analysis_mid_cls[i])))

    print("==="*30)
    print("classification difficult analysis")
    print("==="*30)
    for i in range(10):
        print("classification ({:.1f} ~ {:.1f}):".format(i/10, (i+1)/10))
        for j in range(len(images_labels)):
            print("{:<52s}".format(str(result_analysis_dif_cls[i][j])))
    print("==="*56)


# 初始化输入参数
txt_path = "./dataset/8cls_13k_2k_80k_9k/test.txt"
model_path = "./checkpoint/pth2/TSR_JUSHI_8_mobilenetv2-v0.1_20220313-1409_48_1.0_65_0.9995_0.9968.pth"
save_images_true  = False                                   # 是否保存所有中间的计算图片，针对识别正确的的图片
save_images_false = True                                   # 是否保存所有中间的计算图片，针对识别错误的的图片
path_tmp = "./tmp/result_analyze8new/"  # 中间计算图片存储位置

# 删除现有路径
if os.path.exists(path_tmp): shutil.rmtree(path_tmp)

# 读取图片列表和模型
images_info_list, images_labels = get_images(txt_path)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model = torch.load(model_path)
model.cuda().eval()

# 预先定义用于分析识别结果的存储矩阵
result_analysis_sim_cls = [[0, 0] for _ in range(10)]        # 针对分类网络的数据统计矩阵
result_analysis_mid_cls = [[[0, 0] for _ in range(len(images_labels))] for _ in range(10)]                # 针对分类网络的数据统计矩阵
result_analysis_dif_cls = [[[0]*len(images_labels) for _ in range(len(images_labels))] for _ in range(10)]# 针对分类网络的数据统计矩阵
result = {"simple":result_analysis_sim_cls, "middle":result_analysis_mid_cls, "difficult":result_analysis_dif_cls}

# 循环处理每一张图片
for num, (image_path, label_id) in enumerate(images_info_list):
    image = proc_images(image_path, transform)
    with torch.no_grad():
        output = torch.squeeze(model(image.cuda()).cpu())
        pred_val = torch.softmax(output, dim=0)
        pred_id  = torch.argmax(pred_val).numpy()
        pred_prob = pred_val[pred_id]

        label_id = int(label_id)
        pred_id  = pred_id
        pred_prob = pred_prob.numpy()


    # print(images_labels[label_id], images_labels[pred_id])
    # 分析和存储每一张图片的识别结果
    analyze_result(result_analysis_sim_cls, result_analysis_mid_cls, result_analysis_dif_cls, images_labels, path_tmp, 
                                            image_path, label_id, pred_id, pred_prob, save_images_true, save_images_false)
    if (num+1)%100 == 0: print("finish %d | total %d | percentage %0.2f%%" % (num+1, len(images_info_list), (num+1)/len(images_info_list)*100))

if save_images_false or save_images_true:
    with open(path_tmp + "result.json", "w") as f:
        json.dump(result, f)

# 打印分析结果
print_result(result_analysis_sim_cls, result_analysis_mid_cls, result_analysis_dif_cls, images_labels)
