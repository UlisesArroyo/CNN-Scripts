from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import numpy as np
from time import time
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyperparametros
classes = 6
learning_rate = 0.001
batch_size = 64
num_epoch = 5

name = "modficacion_1"
os.makedirs('./'+"modificaciones", exist_ok=True)


dataset_path = './datasets/cartoon_face/'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'test':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
    'train': 
    datasets.ImageFolder(dataset_path + 'train', data_transforms['train']),
    'test': 
    datasets.ImageFolder(dataset_path + 'test', data_transforms['test'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'test':
    torch.utils.data.DataLoader(image_datasets['test'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

#Se carga el modelo
model = models.resnet50(pretrained=True).to(device)
#Se desabilita el aprendizaje en todas las capas
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 6)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

time1 = time()     

train_accuracy = []
train_loss = []

val_accuracy = []
val_loss = []

def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        file = open("./" + "modificaciones/" + name + ".txt", "a+")
        file.write("Epoch " + str(epoch+1) +"/" + num_epoch + "\n")
        file.close()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        time0 = time()    

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            file = open("./" + "modificaciones/" + name + ".txt", "a+")
            file.write('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)+"\n")
            file.write("time: {:.4f} seg".format(time()-time0)+"\n")
            file.close()
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print("time: {:.4f} seg".format(time()-time0))
            time2 = time()   
            if phase == 'train':
                train_accuracy.append(epoch_acc)
                train_loss.append(loss)
            else:
                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)
        file = open("./" + "modificaciones/" + name + ".txt", "a+")
        file.write("\nTraining Time (in minutes) =",(time2-time1)/60 + "\n")
        file.close()
        print("\nTraining Time (in minutes) =",(time2-time1)/60)

    return model

model_trained = train_model(model, criterion, optimizer, num_epochs)

def getAccuracyLossTrain():
    return train_accuracy, train_loss

def getAccuracyLossVal():
    return val_accuracy, val_loss


epochs = range(1, epochs + 1, 1)


vgg_t_t_a, vgg_t_t_l = getAccuracyLossTrain()

print('Torch train')
print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_t_a), vgg_t_t_a[len(vgg_t_t_a) - 1]))
print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_t_l), vgg_t_t_l[len(vgg_t_t_l) - 1]))

vgg_t_v_a, vgg_t_v_l = getAccuracyLossVal()


print('Torch Val')
print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_v_a), vgg_t_v_a[len(vgg_t_v_a) - 1]))
print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_v_l), vgg_t_v_l[len(vgg_t_v_l) - 1]))
