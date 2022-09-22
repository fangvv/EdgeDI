from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import random


class NEUCLASSDATA():
    def __init__(self):
        self.random_index = []

    # def _get_random_index(self):
    #     index = [0, 300, 600, 900, 1200, 1500, 1800]
    #     for i in range(1, len(index)):
    #         self.random_index += random.sample(range(index[i - 1], index[i]), 90)
    #         # print(self.ramdom_index)

    def _get_data_from_index(self):
        ROOT_DATA = './Data'
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        transform = transforms.Compose(
            [
                #transforms.Resize(200, 200),
                transforms.ToTensor(),
                normalize
            ]
        )
        train_dataset = ImageFolder(ROOT_DATA + "/train_data", transform=transform)
        test_dataset = ImageFolder(ROOT_DATA + "/test_data", transform=transform)
        # self._get_random_index()
        # train_data = []
        # test_data = []
        # for i in range(len(dataset)):
        #     print("NEUCLS Split Running.... %s %%" % (round((i / len(dataset)) * 100, 2)))
        #     if i in self.random_index:
        #         test_data.append(dataset[i])
        #     else:
        #         train_data.append(dataset[i])
        return train_dataset, test_dataset
    

if __name__ == "__main__":
    random_index = random.sample(range(0, 300), 90)
    print(sorted(random_index))