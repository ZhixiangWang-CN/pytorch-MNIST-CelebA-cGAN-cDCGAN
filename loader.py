import torch
import numpy as np
from PIL import Image

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, path,transform=None):
        self.path =path
        self.transform=transform

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        self.data = Image.open(self.path)
        if self.transform is not None:
            self.data = self.transform(self.data)
        self.labels=1
        return self.data, self.labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return 1


if __name__ == '__main__':

    path = './data/lung/1.jpg'
    # 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels
    torch_data = GetLoader(path)
    from torch.utils.data import DataLoader

    # 读取数据
    datas = DataLoader(torch_data, batch_size=1)
    # for i, data in enumerate(datas):
        # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
    for x,y in datas:
        print("x",x)
        print("y", y)