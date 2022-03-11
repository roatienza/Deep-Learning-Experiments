'''
Accelerate demo with fp16 and multi-gpu support.
Single CPU:
    python accelerate_demo.py --cpu

16-bit Floating Point:
    python accelerate_demo.py --fp16

Model from timm:
    python accelerate_demo.py --timm

Singe-GPU:
    python accelerate_demo.py 

Multi-GPU or Multi-CPU:
    accelerate config
    accelerate launch accelerate_demo.py
'''

import torch
import wandb
import datetime
import timm
import torchvision
import argparse
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ui import progress_bar
from accelerate import Accelerator


def init_wandb():
    wandb.login()
    config = {
        "learning_rate": 0.1,
        "epochs": 100,
        "batch_size": 128,
        "dataset": "cifar10"
    }
    run = wandb.init(project="accelerate-options-project", entity="upeee", config=config)

    return run


def run_experiment(args):
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)
    _ = init_wandb()

    # With timm, no need to manually replace the classifier head.
    # Just initialize the model with the correct number of classes.
    # However, timm model has a lower accuracy (TODO: why?)
    if args.timm:
        model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    else:
        model = torchvision.models.resnet18(pretrained=False, progress=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 10) 
    
    # wandb will automatically log the model gradients.
    wandb.watch(model)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=wandb.config.epochs)

    x_train = datasets.CIFAR10(root='./data', train=True, 
                               download=True, 
                               transform=transforms.ToTensor())
    x_test = datasets.CIFAR10(root='./data',
                              train=False, 
                              download=True, 
                              transform=transforms.ToTensor())
    train_loader = DataLoader(x_train, 
                              batch_size=wandb.config.batch_size, 
                              shuffle=True, 
                              num_workers=2)
    test_loader = DataLoader(x_test, 
                             batch_size=wandb.config.batch_size, 
                             shuffle=False, 
                             num_workers=2)



    label_human = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    table_test = wandb.Table(columns=['Image', "Ground Truth", "Initial Pred Label",])

    image, label = iter(test_loader).next()
    image = image.to(accelerator.device)

    # Accelerate API
    model = accelerator.prepare(model)
    optimizer = accelerator.prepare(optimizer)
    scheduler = accelerator.prepare(scheduler)
    train_loader = accelerator.prepare(train_loader)
    test_loader = accelerator.prepare(test_loader)

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(image), dim=1).cpu().numpy()

    for i in range(8):
        table_test.add_data(wandb.Image(image[i]),
                            label_human[label[i]], 
                            label_human[pred[i]])
        accelerator.print(label_human[label[i]], "vs. ",  label_human[pred[i]])



    start_time = datetime.datetime.now()

    best_acc = 0
    for epoch in range(wandb.config["epochs"]):
        train_acc, train_loss = train(epoch, model, optimizer, scheduler, train_loader, loss, accelerator)
        test_acc, test_loss = test(model, test_loader, loss, accelerator)
        if test_acc > best_acc:
            wandb.run.summary["Best accuracy"] = test_acc
            best_acc = test_acc
            if args.fp16:
                accelerator.save(model.state_dict(), "./resnet18_best_acc_fp16.pth")
            else:
                accelerator.save(model, "./resnet18_best_acc.pth")
        wandb.log({
            "Train accuracy": train_acc,
            "Test accuracy": test_acc,
            "Train loss": train_loss,
            "Test loss": test_loss,
            "Learning rate": optimizer.param_groups[0]['lr']
        })

    elapsed_time = datetime.datetime.now() - start_time
    accelerator.print("Elapsed time: %s" % elapsed_time)
    wandb.run.summary["Elapsed train time"] = str(elapsed_time)
    wandb.run.summary["Fp16 enabled"] = str(args.fp16)
    wandb.run.summary["Using timm"] = str(args.timm)

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(image), dim=1).cpu().numpy()

    final_pred = []
    for i in range(8):
        final_pred.append(label_human[pred[i]])
        accelerator.print(label_human[label[i]], "vs. ",  final_pred[i])

    table_test.add_column(name="Final Pred Label", data=final_pred)

    wandb.log({"Test data": table_test})

    wandb.finish()


def train(epoch, model, optimizer, scheduler, train_loader, loss, accelerator):
  model.train()
  train_loss = 0
  correct = 0
  train_samples = 0

  # sample a batch. compute loss and backpropagate
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss_value = loss(output, target)
    accelerator.backward(loss_value)
    optimizer.step()
    scheduler.step(epoch)
    train_loss += loss_value.item()
    train_samples += len(data)
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    if batch_idx % 10 == 0:
      accuracy = 100. * correct / len(train_loader.dataset)
      progress_bar(batch_idx,
                   len(train_loader),
                   'Train Epoch: {}, Loss: {:0.2e}, Acc: {:.2f}%'.format(epoch+1, 
                   train_loss/train_samples, accuracy))
  
  train_loss /= len(train_loader.dataset)
  accuracy = 100. * correct / len(train_loader.dataset)

  return accuracy, train_loss


def test(model, test_loader, loss, accelerator):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:

      output = model(data)
      test_loss += loss(output, target).item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  accuracy = 100. * correct / len(test_loader.dataset)

  accelerator.print('\nTest Loss: {:.4f}, Acc: {:.2f}%\n'.format(test_loss, accuracy))

  return accuracy, test_loss


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--timm", action="store_true", help="If passed, build model using timm library.")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")

    # Seems that this is not supported in the Accelerator version installed
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()