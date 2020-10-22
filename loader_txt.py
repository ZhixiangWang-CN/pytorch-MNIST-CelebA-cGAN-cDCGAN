import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

root = ""


# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            # 移除字符串首尾的换行符
            # 删除末尾空
            # 以空格为分隔符 将字符串分成
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))  # imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transforms.Normalize([0.5], [0.5])
])
train_data = MyDataset(txt=root + 'train.txt', transform=transforms)
test_data = MyDataset(txt=root + 'test.txt', transform=transforms)

# train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2)



for x,y in train_loader:
    print("x",x)
    print("y", y)