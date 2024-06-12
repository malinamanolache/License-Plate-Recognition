from data import RodosolDataset
from torch.utils.data import DataLoader
import sys
import os
import torchvision
import torch
from engine import train_one_epoch, evaluate
import utils

output_dir = "/home/fasterrcnn_models"
batch_size = 2

train_dataset = RodosolDataset(root_dir="/home/datasets/RodoSol-ALPR", split="training")
valid_dataset = RodosolDataset(root_dir="/home/datasets/RodoSol-ALPR", split="validation")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=None, 
                                                                num_classes=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 50

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, valid_loader, device=device)

    model_path = os.path.join(output_dir, f"model_{epoch}.pth")
    torch.save(model.state_dict(), model_path)

print("That's it!")
