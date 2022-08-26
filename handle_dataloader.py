import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

#Gets the number of samples in each class
def get_class_count(dataset):
    class_count = [0] * len(dataset.class_to_idx)
    for _, _class in dataset:
        class_count[_class] += 1
    return class_count

def default_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        #Numbers specified by PyTorch
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def create_dataloader(path, image_transform, batch_size=16, class_rebalance=None):

    dataset = datasets.ImageFolder(path, transform=image_transform)
    generator = torch.Generator()

    if class_rebalance is not None:
        #Random Oversampling
        #Number of samples in each class before oversampling
        class_count = get_class_count(dataset)

        #Calculate new number of samples
        num_samples = 0
        for i in range(len(dataset.class_to_idx)):
            num_samples += int(class_count[i] * class_rebalance[i])

        #Assign weight to each example
        sample_weights = [class_rebalance[dataset[i][1]] for i in range(len(dataset))]

        print(class_rebalance, class_count, num_samples)

        weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

        return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=weighted_sampler, generator=generator)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
