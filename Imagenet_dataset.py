import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImagenetDataset(Dataset):
    def __init__(self, root_path, listfile, image_size=(224, 224), augamentation=False):
        self.image_list = []
        self.label_list = []
        self.image_size = image_size
        self.root_path = root_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(30, translate=(0.3, 0.3), scale=(1.0, 1.5)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize(self.image_size),
            transforms.Normalize(0.5, 0.5),
        ]) if augamentation else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Resize(self.image_size),
            transforms.Normalize(0.5, 0.5),
        ])
        with open(root_path + '/' + listfile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                t = line.split(' ')
                self.image_list.append(t[0])
                self.label_list.append(int(t[1][:-1]))
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.root_path + '/' + self.image_list[idx]).convert('RGB'))

        return img, torch.tensor(self.label_list[idx])

if __name__ == '__main__':
    dataset = ImagenetDataset('/home/SharedDataset/ImageNet', 'val_list.txt')
    print(dataset[100])