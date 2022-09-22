import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from VGG13Block import vgg13Block
from NEUCLSDataLoad import NEUCLASSDATA
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
list_name = []
loss_value = []
acc_value = []
original_layer = [62, 64, 'M', 128, 127, 'M', 256, 255, 'M', 508, 510, 'M', 508, 512, 'M']
model = vgg13Block(original_layer).to(device)
model.eval()
train_data, test_data = NEUCLASSDATA()._get_data_from_index()
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


def _test_model():
    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        for data in test_data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return round((correct / total) * 100, 2)


def _get_name():
    for name in model.state_dict():
        list_name.append(name)


def _load_model():
    path = "./SaveInfo/VGG13/Para/r8PruningPara/"

    for name in list_name:
        temp_load_numpy = np.load(path+"%s.ndim.npy" % (name))
        tensor_load = torch.tensor(temp_load_numpy)
        model.state_dict()[name].copy_(tensor_load)


def _save_result(values):
    path = "./SaveInfo/VGG13/TEST_ACC/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"VGG13-resNet4-test-acc.txt", 'a') as fw:
        for value in values:
            fw.write(str(value) + "-")
        fw.write("\n")
        fw.close()


def _get_acc_value():
    temp_acc = []
    temp_all_acc = 0
    for _ in range(10):
        acc = _test_model()
        temp_all_acc += acc
        temp_acc.append(acc)
    print(temp_acc)
    print("Max_acc --> %s" %(str(max(temp_acc))))
    print("Avg_acc --> %s" %(str(temp_all_acc/10)))
    print("Min_acc --> %s" % (str(min(temp_acc))))
    print("Acc_Error --> %s" %(str(max(temp_acc)-min(temp_acc))))
    _save_result(temp_acc)
    return max(temp_acc), temp_all_acc/10, min(temp_acc)


if __name__ == "__main__":
    _get_name()
    _load_model()
    all_max, all_avg, all_min = 0, 0, 0
    for _ in range(5):
        temp_max, temp_avg, temp_min = _get_acc_value()
        all_max += temp_max
        all_avg += temp_avg
        all_min += temp_min
    print("5 rounds --> max:{}".format(all_max/5))
    print("5 rounds --> avg:{}".format(all_avg / 5))
    print("5 rounds --> min:{}".format(all_min / 5))
    _save_result([all_max/5, all_avg/5, all_min/5])