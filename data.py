import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Data():

    data_dir = 'DividedData'
    torch_seed = 42
    image_size = 227
    batch_size = 32
    shuffle_after_epoch = False

    def load(self):

        torch.manual_seed(self.torch_seed)

        train_transform = transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor()])

        dataset = datasets.ImageFolder(self.data_dir, transform=train_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle_after_epoch)

        images, labels = next(iter(dataloader))
        print(images.size())

        plt.imshow( images[0].squeeze().permute(1,2,0) )
        plt.show()

if __name__ == '__main__':
    d = Data()
    d.load()
