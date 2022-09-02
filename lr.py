import pickle

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder, TrainDataLoaderIter

from data import ShapeNetPart_Augmented
from loss import cross_entropy as criterion
from models.model_partseg import Net

ds = ShapeNetPart_Augmented("trainval", True)
with open("args.pkl", "rb") as f:
    args = pickle.load(f)
model = Net(args)
dl = DataLoader(ds, batch_size=16, pin_memory=True)

class CustomDL(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        data, label, seg = batch_data
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data = data.transpose(2, 1)
        seg = seg.view(-1, 1).squeeze()
        return (data, label_one_hot), seg


trainloader = CustomDL(dl)
optimizer = optim.SGD(model.parameters(), lr=1e-7, weight_decay=0.1, momentum=0.9)
lr_finder = LRFinder(model, optimizer, criterion, device="cuda:3")
lr_finder.range_test(trainloader, end_lr=100, num_iter=1000)
lr_finder.plot()  # to inspect the loss-learning rate graph
lr_finder.reset()  # to reset the model and optimizer to their initial state
