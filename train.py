import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from VGG13 import vgg13
from NEUCLSDataLoad import NEUCLASSDATA
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 20
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
list_name = []
loss_value = []
acc_value = []
original_layer = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
model = vgg13(original_layer).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

train_data, test_data = NEUCLASSDATA()._get_data_from_index()
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


def _train_model():
    max_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch：%s --> lr = %s" % (str(epoch), str(lr)))
        temp_all_loss = 0
        loss_nums = 0
        model.train()
        for data in train_data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels).to(device)
            loss.backward()
            optimizer.step()
            temp_all_loss += loss.item()
            loss_nums += 1
            print("Epoch: %s (AllEpoch: %s) --> loss_value is %s" % (str(epoch), str(NUM_EPOCHS), str(loss.item())))
        epoch_avg_loss = temp_all_loss / loss_nums
        loss_value.append(epoch_avg_loss)
        print("Epoch: %s --> loss_value is % s" % (str(epoch), str(epoch_avg_loss)))
        _adjust_learning_rate(epoch)
        acc = _test_model()
        acc_value.append(acc)
        print("Epoch %s --> acc: %s" %(str(epoch), str(acc)))
        if acc > max_acc:
            _save_model()
            max_acc = acc


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
            print("-------------------------predicted-----------------------")
            print(predicted)
            print("=========================label===========================")
            print(labels)
            total += labels.size(0)
            print("========================correct-sum======================")
            print((predicted == labels).sum().item())
            correct += (predicted == labels).sum().item()
        return round((correct / total) * 100, 2)


def _adjust_learning_rate(epoch):
    if epoch <= 10:
        lr = 0.001
    else:
        lr = 0.0001
    optimizer.param_groups[0]['lr'] = lr


def _get_name():
    for name in model.state_dict():
        list_name.append(name)


def _save_model():
    path = "./SaveInfo/VGG13/Para/Original/"
    if not os.path.exists(path):
        os.makedirs(path)

    for name in list_name:
        temp_np = model.state_dict()[name].cpu().numpy()
        np.save(path+"%s.ndim" % (name), temp_np)

    print("model saved in {}".format(path))


def _save_result():
    path = "./SaveInfo/VGG13/LOSS_ACC/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+"resNet4-lr0001-epoch50-73-loss.txt", 'w') as fw:
        for value in loss_value:
            fw.write(str(value) + "-")
        fw.close()

    with open(path+"resNet4-lr0001-epoch50-73-acc.txt", 'w') as fw:
        for value in acc_value:
            fw.write(str(value)+"-")
        fw.close()


if __name__ == "__main__":
    print(model)
    _get_name()
    _save_model()
    # _train_model()
    # _save_result()