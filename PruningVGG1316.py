import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from VGG13 import vgg13
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
base_min_acc = 98.00
original_layer = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
model = vgg13(original_layer).to(device)
model.eval()
train_data, test_data = NEUCLASSDATA()._get_data_from_index()
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

def _get_Name():
    for name in model.state_dict():
        original_list_name.append(name)


def _load_model():
    path = "./SaveInfo/VGG16/Para/resnet18/"

    for name in original_list_name:
        temp_load_numpy = np.load(path + "%s.ndim.npy" % (name))
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


# #获取当前索引层的名称
def _get_layer_name(index):
    if index < 13:
        start = index * 6
        end = (index + 1) * 6
    else:
        start = index * 6
        end = len(original_list_name)
    feature_name = original_list_name[start:end]
    return feature_name


def _pruning_iter():
    pruning_layer = [65, 63, 'M', 128, 128, 'M', 254, 253, 255, 'M', 507, 512, 511, 'M', 512, 512, 512, 'M']
    layer_sign = [True]*len(pruning_layer)
    for i in range(len(pruning_layer)):
        if pruning_layer[i] == "M": layer_sign[i] = False
    while True:
        for i in range(len(pruning_layer)):
            if layer_sign[i] is False: continue
            pruning_layer[i] = pruning_layer[i] - 1
            pruning_model = vgg13(pruning_layer).to(device)
            print("-------------------------------------------------------------------------------")
            print("Current Layer Infor:")
            print(pruning_layer)
            print("-------------------------------------------------------------------------------")
            for j in range(len(weights_l1)):
                feature_name = _get_layer_name(j)
                for name in feature_name:
                    original_weight_shape = model.state_dict()[feature_name[0]].shape
                    pruning_weight_shape = pruning_model.state_dict()[feature_name[0]].shape
                    axis_0_dis = original_weight_shape[0] - pruning_weight_shape[0]
                    axis_1_dis = original_weight_shape[1] - pruning_weight_shape[1]
                    index_0_pruning = _get_minValue_index(weights_l1[j], axis_0_dis)
                    remove_index[j] = index_0_pruning
                    temp_feature = model.state_dict()[name].cpu()
                    if axis_1_dis > 0 and len(temp_feature.shape) > 1:
                        temp_feature = _resizeFeatureMap(temp_feature, remove_index[j-1], 1)
                    if axis_0_dis > 0 and len(temp_feature.shape) > 0:
                        temp_feature = _resizeFeatureMap(temp_feature, remove_index[j], 0)
                    pruning_model.state_dict()[name].copy_(temp_feature)
            class_name = _get_layer_name(13)
            for name in class_name:
                if len(model.state_dict()[name].shape) > 1:
                    for t in range(model.state_dict()[name].shape[0]):
                        pruning_model.state_dict()[name][t].copy_(model.state_dict()[name][t][:pruning_model.state_dict()[name].shape[1]])
                else:
                    pruning_model.state_dict()[name].copy_(model.state_dict()[name])

            for _ in range(10):
                list_acc.append(_test_model(pruning_model))
            print("---------------------本轮剪枝精度-------------------------")
            print(list_acc)
            pruning_min_acc = min(list_acc)
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


def _resizeFeatureMap(featureMap, index, fm_type):
    featureMap = featureMap.numpy()
    if fm_type == 0:
        featureMap = np.delete(featureMap, obj=index, axis=0)
        return torch.tensor(featureMap)
    else:
        featureMap = np.delete(featureMap, obj=index, axis=1)
        return torch.tensor(featureMap)


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


def _save_model(pruning_model):
    path = "./SaveInfo/VGG16/Para/PruningModel/"
    if not os.path.exists(path):
        os.makedirs(path)

    for name in original_list_name:
        temp_np = pruning_model.state_dict()[name].cpu().numpy()
        np.save(path + "%s.ndim" % (name), temp_np)


def _saveLayer(layer_info, layer_name):
    with open("./SaveResult/PruningResult/"+str(layer_name)+".txt", 'w') as fw:
        for layer in layer_info:
            fw.write(str(layer)+"-")
        fw.close()


if __name__ == "__main__":
    _get_Name()
    _load_model()
    _computer_L1_value()
    _pruning_iter()