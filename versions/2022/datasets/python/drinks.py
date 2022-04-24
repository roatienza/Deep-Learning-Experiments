
import torch
import numpy as np
import label_utils
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")
train_dict, train_classes = label_utils.build_label_dictionary("drinks/labels_train.csv")


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transform=None):
        self.dictionary = dictionary
        self.transform = transform

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
        # retrieve all bounding boxes
        boxes = self.dictionary[key]
        # open the file as a PIL image
        img = Image.open(key)
        # apply the necessary transforms
        # transforms like crop, resize, normalize, etc
        if self.transform:
            img = self.transform(img)
        
        # return a list of images and corresponding labels
        return img, boxes


train_split = ImageDataset(train_dict, transforms.ToTensor())
test_split = ImageDataset(test_dict, transforms.ToTensor())

# This is approx 95/5 split
print("Train split len:", len(train_split))
print("Test split len:", len(test_split))

# We do not have a validation split

def collate_fn(batch):
    maxlen = max([len(x[1]) for x in batch])
    images = []
    boxes = []
    for i in range(len(batch)):
        img, box = batch[i]
        images.append(img)
        # pad with zeros if less than maxlen
        if len(box) < maxlen:
            box = np.concatenate(
                (box, np.zeros((maxlen-len(box), box.shape[-1]))), axis=0)

        box = torch.from_numpy(box)
        boxes.append(box)

    return torch.stack(images, 0), torch.stack(boxes, 0)


train_loader = DataLoader(train_split,
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=config['num_workers'],
                          pin_memory=config['pin_memory'],
                          collate_fn=collate_fn)

test_loader = DataLoader(test_split,
                         batch_size=config['batch_size'],
                         shuffle=False,
                         num_workers=config['num_workers'],
                         pin_memory=config['pin_memory'],
                         collate_fn=collate_fn)