import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class Resnet_ImagenetDataset(Dataset):
    def __init__(self, root_path, listfile, image_size=(224, 224), augamentation=False, dataset_type = "train"):
        self.image_list = []
        self.label_list = []
        self.image_size = image_size
        self.root_path = root_path
        
        self.transform = transforms.Compose([
            transforms.Resize(256),              # Resize the image to 256x256
            transforms.RandomCrop(224),          # Random crop to 224x224
            transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            
            transforms.RandomAffine(30, translate=(0.3, 0.3), scale=(1.0, 1.5)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize(self.image_size),

            transforms.ToTensor(),               # Convert image to PyTorch tensor
            transforms.Normalize(                # Normalize using ImageNet mean and std
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ]) if augamentation else transforms.Compose([
            transforms.Resize(256),              # Resize the image to 256x256
            transforms.CenterCrop(224),          # Random crop to 224x224
            transforms.RandomHorizontalFlip(),   # Randomly flip the image horizontally
            transforms.ToTensor(),               # Convert image to PyTorch tensor
            transforms.Normalize(                # Normalize using ImageNet mean and std
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])



        with open(root_path + '/' + listfile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                t = line.split(' ')
                self.image_list.append(t[0])
                self.label_list.append(int(t[1][:-1]))
            # for line, idx in zip(lines, range(len(lines))):
            #     t = line.split(' ')
            #     self.image_list.append(t[0])
            #     self.label_list.append(int(t[1][:-1]))
            #     if idx >= 100:  break
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.root_path + '/' + self.image_list[idx]).convert('RGB'))

        return img, torch.tensor(self.label_list[idx])

if __name__ == '__main__':
    dataset = Resnet_ImagenetDataset('/home/SharedDataset/ImageNet', 'val_list.txt')
    print(dataset[100])