import numpy as np
import torch 
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import shutil
import argparse
import os


def loading_data(data_dir):
    train_dir= os.path.join(data_dir,"train")
    valid_dir= os.path.join(data_dir,"valid")
    
    train_transforms= transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    valid_transforms= transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_dataset=datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset= datasets.ImageFolder(valid_dir, transform=valid_transforms)
    train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader=torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    
    return train_dataloader, valid_dataloader, train_dataset.class_to_idx


def build_model(arch="vgg16", hidden_layers=4096, class_idx_mapping=None):
    locally= dict()
    exec("model= models.{}(pretrained=True)".format(arch),globals(), locally)
    
    model=locally['model']
    last_features= list(model.children())[-1]
    if type(last_features)==torch.nn.modules.linear.Linear:
        input_features= last_features.in_features
    elif type(last_features)== torch.nn.modules.container.Sequential:
        input_features= last_features[0].in_features
        
    for param in model.parameters():
        param.requires_grad=False
    classifier=nn.Sequential(nn.Linear(input_features, hidden_layers),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(hidden_layers,102),
                            nn.LogSoftmax(dim=1))
    model.classifier= classifier
    model.class_idx_mapping= class_idx_mapping
    return model

def train(model, trainloader, validloader, epochs, criterion, optimizer, arch="vgg16", device="cuda", model_dir="models"):
    model.train()
    model.to(device)
    best_accuracy=0
    epochs=epochs
    train_losses, test_losses = [], []
    
    for epoch  in range(epochs):
        train_loss=0
        for images, labels in trainloader:
            images, labels= images.to(device), labels.to(device)

            optimizer.zero_grad()
            output= model(images)
            loss=criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss+= loss.item()
        test_loss=0
        accuracy=0
        with torch.no_grad():
            model.eval()
            for images, labels in validloader:
                images, labels= images.to(device), labels.to(device)
                output= model(images)
                loss=criterion(output, labels)
                test_loss+= loss.item()
                ps= torch.exp(output)
                top_p, top_class= ps.topk(1, dim=1)
                equals= top_class==labels.view(*top_class.shape)
                accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
            model.train()
        train_losses.append(train_loss/len(trainloader))
        test_losses.append(test_loss/len(validloader))

        print("Epoch: {}/{}".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(trainloader)),
             "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
             "Validation accuracy: {:.3f}".format((accuracy/len(validloader))*100))
        
        is_best= accuracy > best_accuracy
        best_accuracy= max(accuracy, best_accuracy)
        save_checkpoint({'epochs': epochs,
                        'classifier': model.classifier,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'class_idx_mapping': model.class_idx_mapping,
                        'arch': arch,
                        'best_accuracy': (best_accuracy/len(validloader))*100
                        }, is_best, model_dir, 'checkpoint.pth')
                
def save_checkpoint(state, is_best=False, model_dir='models', filename='checkpoint.pth'):
    torch.save(state, os.path.join(model_dir, filename))
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))

    
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Directory containing the dataset.",
                    default="data", nargs="?")

    VALID_ARCH_CHOICES = ("vgg16", "vgg13", "densenet121")
    ap.add_argument("--arch", help="Model architecture from 'torchvision.models'. (default: vgg16)", choices=VALID_ARCH_CHOICES,
                    default=VALID_ARCH_CHOICES[0])

    ap.add_argument("--hidden_units", help="Number of units the hidden layer should consist of. (default: 4096)",
                    default=4096, type=int)

    ap.add_argument("--learning_rate", help="Learning rate for Adam optimizer. (default: 0.001)",
                    default=0.001, type=float)

    ap.add_argument("--epochs", help="Number of iterations over the whole dataset. (default: 3)",
                    default=3, type=int)

    ap.add_argument("--gpu", help="Use GPU or CPU for training",
                    action="store_true")

    ap.add_argument("--model_dir", help="Directory which will contain the model checkpoints.",
                    default="models")
    args = vars(ap.parse_args())

    os.system("mkdir -p " + args["model_dir"])

    (train_dataloader, valid_dataloader, class_idx_mapping) = loading_data(data_dir=args["data_dir"])
    
    model = build_model(arch=args["arch"], hidden_layers=args["hidden_units"], class_idx_mapping=class_idx_mapping)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args["learning_rate"])

    device = None
    if args["gpu"]:
        device = "cuda"
    else:
        device = "cpu"

    train(model=model, 
        trainloader=train_dataloader, 
        validloader=valid_dataloader,
        epochs=args["epochs"],
        criterion=criterion,
        optimizer=optimizer,
        arch=args["arch"],
        device=device,
        model_dir=args["model_dir"])

if __name__ == '__main__':
    main()
    