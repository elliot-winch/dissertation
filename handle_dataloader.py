import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def create_dataloader(path, image_transform, batch_size=16, use_sampling=False, class_balance=None):

    dataset = datasets.ImageFolder(path, transform=image_transform)

    if use_sampling and class_balance is not None:
        class_count = get_class_count(dataset)

        #Class with the most examples
        max_class_count = max(class_count)

        #The number of examples in one class would be multiplied by
        #its class weight to match the number of examples in the
        #majority class
        upsample_factor = [max_class_count / i for i in class_count]

        #Calculate total number of samples
        num_samples = 0
        for i in range(0, len(dataset.class_to_idx)):
            upsample_factor[i] *= class_balance[i]
            num_samples += int(class_count[i] * upsample_factor[i])

        #Assign weight to each example
        example_weights = [0] * len(dataset)
        for i in range(0, len(dataset)):
            example_weights[i] = upsample_factor[dataset[i][1]]

        weighted_sampler = WeightedRandomSampler(weights=example_weights, num_samples=num_samples, replacement=True)

        print("Upsampling results: \nClass_count: {}\nUpsample factor: {}\nNum samples: {}\nExample weights: {}".format, class_count, upsample_factor, num_samples, example_weights)

        return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=weighted_sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
