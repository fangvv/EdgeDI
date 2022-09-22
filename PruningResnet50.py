import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ResNet50 import resnet50
from NEUCLSDataLoad import NEUCLASSDATA
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
LEARNING_RATE = 0.001
original_list_name = []
weights_l1 = []
list_acc = []
remove_index = {}
base_min_acc = 96.67
original_layer = [64, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
model = resnet50(original_layer).to(device)
model.eval()

train_data, test_data = NEUCLASSDATA()._get_data_from_index()
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


def _get_Name():
    for name in model.state_dict():
        original_list_name.append(name)


def _load_model():
    for name in original_list_name:
        temp_load_numpy = np.load("./SaveInfo/ResNet50/Para/resnet18/%s.ndim.npy" % (name))
        tensor_load = torch.tensor(temp_load_numpy)
        model.state_dict()[name].copy_(tensor_load)


def _computer_L1_value():
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            weights_copy = module.weight.data.abs().clone()
            weights_copy = weights_copy.cpu().numpy()
            weights_copy_l1 = np.sum(weights_copy, axis=(1, 2, 3))
            weights_l1.append(weights_copy_l1.tolist())


#寻找层中最N个最小值的坐标
def _get_minValue_index(value, nums):
    value_sort = np.argsort(value)
    index = value_sort[:nums]
    return index


#获取当前索引层的名称
def _get_layer_name(index):
    name_sign = 'feature.' + str(index)+"."
    feature_name = []
    for name in original_list_name:
        if name_sign in name:
            feature_name.append(name)
    if feature_name != []:
        return feature_name
    else:
        return original_list_name[len(original_list_name)-2:]


def _resizeFeatureMap(featureMap, index, fm_type):
    featureMap = featureMap.numpy()
    if fm_type == 0:
        # print("--------fmtype0---------------")
        # print(featureMap.shape)
        # print(index)
        featureMap = np.delete(featureMap, obj=index, axis=0)
        # print(featureMap.shape)
        return torch.tensor(featureMap)
    else:
        # print("--------fmtype1---------------")
        # print(featureMap.shape)
        # print(index)
        featureMap = np.delete(featureMap, obj=index, axis=1)
        # print(featureMap.shape)
        return torch.tensor(featureMap)
    

def _resizeFeatureMap_axis1_self(featureMap, nums):
    # print("===============self_axis1_prun======================")
    featureMap = featureMap.numpy()
    featureMap_list = []
    weight_index = np.sum(featureMap, axis=(2, 3))
    for i in range(featureMap.shape[0]):
        index = np.argsort(weight_index[i])
        temp = np.delete(featureMap[i], index[:nums], axis=0)
        featureMap_list.append(temp.tolist())
    return torch.tensor(np.array(featureMap_list))


def _resizeFeatureMap_axis0_self(featureMap, nums):
    # print("===============self_axis0_prun======================")
    featureMap = featureMap.numpy()
    if len(featureMap.shape) == 4:
        weight_index = np.sum(featureMap, axis=(1, 2, 3))
        index = np.argsort(weight_index)
        featureMap = np.delete(featureMap, index[:nums], axis=0)
        return torch.tensor(featureMap)
    else:
        index = np.argsort(featureMap)
        featureMap = np.delete(featureMap, index[:nums], axis=0)
        return torch.tensor(featureMap)
        

def _pruning_iter():
    pruning_layer = [65, 256, 256, 256, 512, 512, 512, 512, 1024, 1022, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
    layer_sign = [True]*len(pruning_layer)
    while True:
        for i in range(len(pruning_layer)):
            if layer_sign[i] is False: continue
            pruning_layer[i] = pruning_layer[i] - 1
            pruning_model = resnet50(pruning_layer).to(device)
            print("-------------------------------------------------------------------------------")
            print("Current Layer Infor:")
            print(pruning_layer)
            print("-------------------------------------------------------------------------------")
            for j in range(len(weights_l1)):
                feature_name = _get_layer_name(j)
                # print(feature_name)
                sign_name = feature_name[0]
                for name in feature_name:
                    # print("name:" + str(name))
                    original_weight_shape = model.state_dict()[sign_name].shape
                    pruning_weight_shape = pruning_model.state_dict()[sign_name].shape
                    # print(model.state_dict()[name].shape)
                    # print(pruning_model.state_dict()[name].shape)
                    if len(model.state_dict()[name].shape) == 4:
                        sign_name = name
                        original_weight_shape = model.state_dict()[sign_name].shape
                        pruning_weight_shape = pruning_model.state_dict()[sign_name].shape
                    axis_0_dis = original_weight_shape[0] - pruning_weight_shape[0]
                    axis_1_dis = original_weight_shape[1] - pruning_weight_shape[1]
                    # print(original_weight_shape)
                    # print(pruning_weight_shape)
                    # print("============================================================")
                    # print(axis_0_dis)
                    # print(axis_1_dis)
                    axis_1_dis_index, axis_0_dis_index = [], []
                    if axis_1_dis > 0:
                        axis_1_dis_index = _get_minValue_index(weights_l1[j-1], axis_1_dis)
                    if axis_0_dis > 0:
                        axis_0_dis_index = _get_minValue_index(weights_l1[j], axis_0_dis)
                    # print(axis_0_dis_index)
                    # print(axis_1_dis_index)
                    temp_feature = model.state_dict()[name].cpu()
                    if axis_1_dis > 0 and len(temp_feature.shape) > 1:
                        if max(axis_1_dis_index) < temp_feature.shape[1]:
                            temp_feature = _resizeFeatureMap(temp_feature, axis_1_dis_index, 1)
                        else:
                            temp_feature = _resizeFeatureMap_axis1_self(temp_feature, axis_1_dis)
                    # print(temp_feature.shape)
                    if axis_0_dis > 0 and len(temp_feature.shape) > 0:
                        if max(axis_0_dis_index) < temp_feature.shape[0]:
                            temp_feature = _resizeFeatureMap(temp_feature, axis_0_dis_index, 0)
                        else:
                            temp_feature = _resizeFeatureMap_axis0_self(temp_feature, axis_0_dis)
                    # print(temp_feature.shape)
                    pruning_model.state_dict()[name].copy_(temp_feature)
            class_name = _get_layer_name(17)
            for name in class_name:
                if len(model.state_dict()[name].shape) > 1:
                    for t in range(model.state_dict()[name].shape[0]):
                        pruning_model.state_dict()[name][t].copy_(
                            model.state_dict()[name][t][:pruning_model.state_dict()[name].shape[1]])
                else:
                    pruning_model.state_dict()[name].copy_(model.state_dict()[name])
            for o in range(10):
                list_acc.append(_test_model(pruning_model))
            print("---------------------本轮剪枝精度-------------------------")
            print(list_acc)
            # pruning_min_acc = min(list_acc)
            # list_acc.clear()
            # save_sign = True
            # print("本轮 min 精度：" + str(pruning_min_acc) + "%")
            # if pruning_min_acc < base_min_acc:
            #     layer_sign[i] = False
            #     pruning_layer[i] = pruning_layer[i] + 1
            #     save_sign = False
            # if save_sign:
            #     _save_model(pruning_model)
            _save_model(pruning_model)
            break
        # print(layer_sign)
        # if True not in layer_sign:
        #     print("Goal Layer Numbers: " + str(pruning_layer))
        #     _saveLayer(pruning_layer, 'pruning_layer_info_iter1')
        #     break
        break


def _save_model(pruning_model):
    path = "./SaveInfo/ResNet50/Para/PruningModel/"
    if not os.path.exists(path):
        os.makedirs(path)

    for name in original_list_name:
        temp_np = pruning_model.state_dict()[name].cpu().numpy()
        np.save(path + "%s.ndim" % (name), temp_np)


def _saveLayer(layer_info, layer_name):
    with open("./SaveResult/PruningResult-Res50/"+str(layer_name)+".txt", 'w') as fw:
        for layer in layer_info:
            fw.write(str(layer)+"-")
        fw.close()


def _test_model(pruning_model):
    with torch.no_grad():
        total = 0
        correct = 0
        for data in test_data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = pruning_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return round((correct / total) * 100, 2)


if __name__ == "__main__":
    _get_Name()
    _load_model()
    _computer_L1_value()
    _pruning_iter()