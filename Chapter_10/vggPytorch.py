from traceback import format_exc
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader
from torchvision.utils import save_image
import torch.optim as optim
import torchvision

#Selección quien procesara la chamaba la gpu o el cpu 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#Direcciones de nuestra dataset
train_path = "./datasets/cartoon_face/train"
test_path = "./datasets/cartoon_face/test"

#Modificaciones a los datasets

tranform_train = transforms.Compose([transforms.Resize((224,224)), #Normaliza el tamaño de las imagenes (eso creo)
                                    transforms.RandomHorizontalFlip(p=0.1), #la voltean o algo asi, pero sigue siendo una imagen como la conocemos
                                    transforms.ToTensor(), #Lo volvemos un tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], #Esto no sé que hace xD
                                    std=[0.229, 0.224, 0.225])])#x2

tranform_test = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.RandomHorizontalFlip(p=0.1),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])

#preparing the train, validation and test dataset
#Inicializamos nuestros datasets como dios manda
train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform= tranform_train)
test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform= tranform_train)
#La longitud es la cantidad de imagenes
print("Longitud de train_data"len(train_dataset)) #Este ejemplo son 1800

train_dataloader = DataLoader(train_dataset, batch_size= 16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size= 16, shuffle=False)



class VGGTorch(nn.Module):#Se repite la estructura del modelo, por que? cual se necesitan los dos?
    def __init__(self) -> None:
        super(VGGTorch, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25088, 4096) #El 25088 puede ser el resultado de 512 * 7 * 7, se redujo las dimensiones en las convs
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 6) 

    
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1) # Este actua como el Flatten
        #x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        #x = F.relu(self.fc2(x))
        #x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x

class VggNetTorch():
    def __init__(self, learning_rate, epochs, device):
        self.model = VGGTorch().to(device) #Lo mismo que arriba pero mas barato

        self.criterion = nn.CrossEntropyLoss()#Esto es la funcion de perdida? Investigar
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) #Que es un optimizador? que otro hay aparte del Adam? Investigar

        self.epochs = epochs #Se pasa el valor de las epocas
        self.device = device #Se pasa el dato si se utiliza la gpu o la cpu
        #La documentación de pytorch no ocupa el device asi

    def getAccuracyLossTrain(self):
        return self.trainAccuracy, self.trainLoss

    def getAccuracyLossVal(self):
        return self.valAccuracy, self.valLoss

    def train(self):
        self.model = self.model.cuda(device=self.device)#Aqui ya doy el ultimo vobo para elegir la gpu?
        print("Se utilizara: " + self.device)

        self.trainAccuracy = []
        self.trainLoss = []

        self.valAccuracy = []
        self.valLoss = []

        for epoch in range(self.epochs): #I decided to train the model for 50 epochs
            loss_ep = 0
            
            num_correct = 0
            num_samples = 0
            for batch_idx, (data, targets) in enumerate(train_dataloader):
                data = data.to(device=self.device)#Se carga el data al gpu
                targets = targets.to(device=self.device)
                ## Forward Pass
                self.optimizer.zero_grad()
                
                scores = self.model(data)
                _, predictions = scores.max(1)

                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
                
                loss = self.criterion(scores,targets)
                loss.backward()
                
                self.optimizer.step()
                loss_ep += loss.item()
            
            print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dataloader)}")
            self.trainAccuracy.append(float(num_correct) / float(num_samples))
            self.trainLoss.append(loss_ep/len(train_dataloader))

            with torch.no_grad():
                loss_ep = 0
                num_correct = 0
                num_samples = 0
                for batch_idx, (data,targets) in enumerate(test_dataloader):
                    data = data.to(device=self.device)
                    targets = targets.to(device=self.device)
                    ## Forward Pass
                    scores = self.model(data)
                    _, predictions = scores.max(1)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)

                    loss = self.criterion(scores,targets)
                    loss_ep += loss.item()

                print(
                    f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
                )
                self.valAccuracy.append(float(num_correct) / float(num_samples))
                self.valLoss.append(loss_ep/len(test_dataloader)) 

if __name__ == "__main__":
    #Los hiperparametros
    epochs = 30
    batch_size = 16
    learning_rate = 0.0001
    
    #Contiene la estructura de la red vista el libro y cosas extras que en keras no tiene.
    vgg_torch = VggNetTorch(learning_rate, epochs, device)
    vgg_torch.train()
    
    vgg_t_t_a, vgg_t_t_l = vgg_torch.getAccuracyLossTrain()

    print('Torch train')
    print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_t_a), vgg_t_t_a[len(vgg_t_t_a) - 1]))
    print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_t_l), vgg_t_t_l[len(vgg_t_t_l) - 1]))

    vgg_t_v_a, vgg_t_v_l = vgg_torch.getAccuracyLossVal()


    print('Torch Val')
    print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_v_a), vgg_t_v_a[len(vgg_t_v_a) - 1]))
    print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_v_l), vgg_t_v_l[len(vgg_t_v_l) - 1]))

    plt.plot ( epochs, vgg_t_t_a, 'g--', label='Training acc Torch'  )
    plt.plot ( epochs, vgg_t_v_a,  'c', label='Validation acc Torch')
    plt.title ('Training and validation accuracy Keras - Torch')
    plt.ylabel('acc')
    plt.xlabel('epochs')

    plt.legend()
    plt.figure()
    plt.show()

    plt.plot ( epochs, vgg_t_t_l, 'g--', label='Training loss Torch'  )
    plt.plot ( epochs, vgg_t_v_l,  'c', label='Validation loss Torch')

    plt.title ('Training and validation loss Keras - Torch')
    plt.ylabel('loss')
    plt.xlabel('epochs')

    plt.legend()
    plt.figure()
    plt.show()



