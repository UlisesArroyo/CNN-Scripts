#Librerias y asociados 
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import numpy as np
import time
import os

#GPU o CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyperparametros
classes = 6
learning_rate = 0.001
batch_size = 64
num_epoch = 5


dataset_path = './datasets/cartoon_face/'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#Esta normalizacion de que va?

#Normalizacion los datasets
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
#SE cargan los datasets modificados
dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,#Cantidad de imagenes en cada lote
                                shuffle=True,
                                num_workers=0),  # for Kaggle
    'test':
    torch.utils.data.DataLoader(image_datasets['test'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=0)  # for Kaggle
}

#En este caso el dataset se pide de torchvision
model = models.resnet50(pretrained=True).to(device)
    
for param in model.parameters():#Recorre los parametros del modelo y desabilita para que no aprendan, aqui esta la clave
    param.requires_grad = True 
    
model.fc = nn.Sequential(#Me parece que esta sustituyendo la ultima capa de la red por esta wea
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 6)).to(device)

#Hiperparametros
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

train_accuracy = []
train_loss = []

val_accuracy = []
val_loss = []

def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

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

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_accuracy.append(epoch_acc)
                train_loss.append(loss)
            else:
                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)

    return model

model_trained = train_model(model, criterion, optimizer, num_epochs=30)

def getAccuracyLossTrain():
    return train_accuracy, train_loss

def getAccuracyLossVal():
    return val_accuracy, val_loss


if __name__ == "__main__":
    epochs = 30
    batch_size = 64

    epochs = range(1, epochs + 1, 1)

    vgg_t_t_a, vgg_t_t_l = getAccuracyLossTrain()

    print('Torch train')
    print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_t_a), vgg_t_t_a[len(vgg_t_t_a) - 1]))
    print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_t_l), vgg_t_t_l[len(vgg_t_t_l) - 1]))

    vgg_t_v_a, vgg_t_v_l = getAccuracyLossVal()


    print('Torch Val')
    print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_v_a), vgg_t_v_a[len(vgg_t_v_a) - 1]))
    print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_v_l), vgg_t_v_l[len(vgg_t_v_l) - 1]))



"""
    plt.plot ( epochs, vgg_t_t_l, 'g--', label='Training loss Torch'  )
    plt.plot ( epochs, vgg_t_v_l,  'c', label='Validation loss Torch')

    plt.title ('Training and validation loss Keras - Torch')
    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.legend()
    plt.figure()
    plt.show()
"""
