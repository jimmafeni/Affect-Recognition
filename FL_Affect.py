
# coding: utf-8

# In[191]:


import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import torch.utils.data.dataloader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification,classification_report,confusion_matrix,accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
#from torchsummary import summary
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut,KFold,StratifiedKFold
import time
import os
import copy
from sklearn import preprocessing
from PIL import Image
from torch.optim import lr_scheduler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# In[620]:


# dir_videotiming = '../RECOLA-Video-timings/RECOLA-Video-timings/P16.csv'
dir_video = 'folder containing images'
arousal_file = 'groudtruth folder for arousal'
valence_file = 'groudtruth folder for valence'
image_names = 'file containing names of images'
part_file = 'file containing names of participants'
log_file = 'log file'


# Face extractor


mtcnn = MTCNN(image_size=224,min_face_size=10, thresholds=[0.5, 0.5, 0.5])

# Load participant file


df_part = pd.read_csv(part_file)


# Results output function




def print_dropped(md,feat,sql,hs,nl,part,mse,mse2,mae,mae2,
                  rmse, rmse2,ccc,ccc2,file=log_file ):    
    pd.DataFrame({'a':[md],'n':[feat],'b':[sql],'c':[hs],'d':[nl],
                  'e':[part],'f':[mse],'g':[mse2],'h':[mae],'i':[mae2],
                  'j':[rmse], 'k':[rmse2],'l':[ccc],'m':[ccc2]}).to_csv(file, mode='a', header=False)  
    print('Added to output file')    


# Performance evaluation metrics

# In[633]:


def merror(true,pred):
    d=np.abs(pred.detach().cpu().numpy()-true.detach().cpu().numpy())
    length = len(d)
    err = 0
    for i in range(length):
        err += d[i]
    err = (err/length)    
    #err=np.sum(d)/length
    return err


# In[634]:


def merror1(true,pred):
    d=np.abs(pred-true)
    length = len(d)
    err=np.sum(d)/length
    return err


# In[635]:


def mse(true, pred):
    d_squared = (pred.detach().cpu().numpy() - true.detach().cpu().numpy())**2
    length = len(d_squared)
    err = 0
    for i in range(length):
        err += d_squared[i]
    err = (err/length)
    return err

def mse1(true, pred):
    d_squared = (pred - true)**2
    length = len(d_squared)
    err = 0
    for i in range(length):
        err += d_squared[i]
    err = (err/length)
    return err


# In[637]:


def rmse(true, pred):
    d_squared = (pred.detach().cpu().numpy()-true.detach().cpu().numpy())**2
    length = len(d_squared)
    err = 0
    for i in range(length):
        err += d_squared[i]
    err = np.sqrt(err/length)
    return err


# In[638]:


def rmse1(true, pred):
    d_squared = (pred - true)**2
    length = len(d_squared)
    err = 0
    for i in range(length):
        err += d_squared[i]
    err = np.sqrt(err/length)
    return err


# In[639]:


#concordance_correlation_coefficient
def ccc(y_true, y_pred):
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator


# Aggregate annotation

# In[640]:


def aggre_annot(annot):
    
    return round(annot.iloc[:,1:].mean(axis=1),4)   


# Dataloader function



# Normalisation
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = self.transform(image)
        #image = Image.open(img_path)
        #print(img_path)
        #image = image.convert('RGB')
        #image = mtcnn(image)
        #print(type(image))
        #if image is None:
            #image = read_image(img_path)
            #image = self.transform(image)

        label = torch.tensor(self.img_labels.iloc[idx, 1:])
        #if self.transform:
            #image = self.transform(image)

        return image, label

#extract each participants data and store in a dataloader: X->features Y-> 2dimensional(valence and arousal)
def extract_data(df_part, dir_video, image_names , arousal_file, valence_file, batch_size):
    #list to store dataloaders
    list_dataloaders=[]
    
    for i in range(len(df_part)):
    #for i in range(4):
        #extract participant code from participants.csv file
        partFile = df_part.Participants[i]
        #valence csv
        df_valence = pd.read_csv(valence_file + partFile +'.csv',sep=';')
        #arousal csv
        df_arousal = pd.read_csv(arousal_file + partFile+'.csv',sep=';')
        df_valence = aggre_annot(df_valence)
        df_arousal = aggre_annot(df_arousal)
        #get image names
        df_images = pd.read_csv(image_names + partFile+'.csv',sep=';')
        #merge valence and arousal -> Y
        Y = pd.DataFrame({'valence':df_valence,'arousal':df_arousal})
        Y = pd.concat([df_images, Y], axis=1)
        #Y = Y[:1000]
        #creat dataloaders
        train_data = CustomImageDataset(Y, dir_video + partFile,transform)
        dataloaders = torch.utils.data.DataLoader(train_data, batch_size = batch_size,shuffle = True,drop_last = False, num_workers=4) #low =True, high=False       
        #add dataloader to the list of dataloaders
        list_dataloaders.append(dataloaders)
    
    return list_dataloaders
    


# Set model parameters

# pick a window size of 8: used 8 for 8 hours of the trading
seq_len = 8

# batch_size
batch_size = 16

# number of features
input_features = 64

# number of hidden units
hidden_size = 1

#number lstm layers
num_layers = 1

#output dimension
output_dim = 2

#learning rate
lr = 0.0001

#number of epochs
num_epochs = 10

mdl = 'CNN_GRUFL_50'

#model file
file = "CNN_False.pth"

#MODELS

def CNN():
    
    preNet = models.resnet18(pretrained=True)    

    preNet.fc = nn.Sequential()


    return preNet

class RNN_model(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim,seq_len):
        super(RNN_model, self).__init__()
        
        self.cnn = CNN()
        
               
        self.rnn = nn.RNN(input_features, hidden_size, num_layers=num_layers)
        
        self.out1 = nn.Linear(hidden_size*seq_len, output_dim)  
        
        #self.out2 = nn.Linear(10, output_dim) 
        

        
    def forward(self, input):
        


        input1 = self.cnn(input)

               
        output, hidden = self.rnn(input1.view(seq_len,-1,input_features))
        
        
        pred = self.out1(output.view(-1,seq_len*hidden_size))     

             
        #pred = self.out2(self.out1(output.view(-1,hidden_size*2)))

        return pred




class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim,seq_len):
        super(BiLSTM, self).__init__()
        
        self.cnn = CNN()
        
               
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers=num_layers, bidirectional=True)
        
        self.out1 = nn.Linear(hidden_size*2*seq_len, output_dim)  
        
        #self.out2 = nn.Linear(10, output_dim) 
        

        
    def forward(self, input):
        


        input1 = self.cnn(input)

               
        output, hidden = self.lstm(input1.view(seq_len,-1,input_features))
        
        
        pred = self.out1(output.view(-1,seq_len*hidden_size*2))     

             
        #pred = self.out2(self.out1(output.view(-1,hidden_size*2)))

        return pred





#BiGRU classification model
class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim,seq_len):
        super(BiGRU, self).__init__()
        
        self.cnn = CNN()
        
               
        self.gru = nn.GRU(input_features, hidden_size, num_layers=num_layers, bidirectional=True)
        
        self.out1 = nn.Linear(hidden_size*2*seq_len, output_dim)  
        
        #self.out2 = nn.Linear(10, output_dim) 
        

        
    def forward(self, input):
        


        input1 = self.cnn(input)

               
        output, hidden = self.gru(input1.view(seq_len,-1,input_features))
        
        
        pred = self.out1(output.view(-1,seq_len*hidden_size*2))     

             
        #pred = self.out2(self.out1(output.view(-1,hidden_size*2)))

        return pred


list_dataloaders = extract_data(df_part, dir_video, image_names, arousal_file, valence_file, batch_size)


#Client update function

def client_update_loss(client_model,loss_client, optimizer, train_loader, epoch=2):
    """
    This function updates/trains client model on client data
    """

    use_cuda = True
  
    client_model.train()
    for e in range(epoch):
        for data, target in train_loader:
            if use_cuda and torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            output = client_model(data.float())
            loss = loss_client(output, target.float())          
            loss.backward()
            optimizer.step()
    return loss.item()


# Initialise model



def initialise_model(input_features, hidden_size,num_layers,output_dim,lr,seq_len):
    
    model = BiGRU(input_features, hidden_size,num_layers,output_dim,seq_len)

    #model = CNN(output_dim)
    
    use_cuda = True

    if use_cuda and torch.cuda.is_available():
        model.cuda()


    return model


# Training function





def train_network(dataloaders_list, list_part, epochs, file, input_features, hidden_size,num_layers,output_dim,lr,seq_len):
    use_cuda = True
    
    since = time.time()
    
    list_models = []    
    
    loss_cnn = nn.MSELoss()
    
    #global model
    
    
    k = KFold(8)
    num_loo = 1
    for train_index, test_index in k.split(dataloaders_list):  
        train_history = []
    
        val_history = []
        
        global_model = initialise_model(input_features, hidden_size,num_layers,output_dim,lr,seq_len)
    
        best_model_wts = copy.deepcopy(global_model.state_dict())

        valence_best_error = 50.0
        arousal_best_error = 50.0
        best_loss = 50.0
    
        #leave-one-out cross-validation
        for epoch in range(epochs):
            
            running_loss = 0.0 

            client_models = [initialise_model(input_features, hidden_size,num_layers,output_dim,lr,seq_len)
                             for _ in range(len(train_index))]   

            for model in client_models:
                model.load_state_dict(global_model.state_dict())

            opt = [optim.Adam(model.parameters(), lr=lr) for model in client_models]


            for i in range(len(train_index)):
                running_loss += client_update_loss(client_models[i],loss_cnn, opt[i], dataloaders_list[train_index[i]], epoch=5)
                
            train_history.append(running_loss)
            
            print('Epoch[%d] LOO[%d]  Train loss: %.8f'  % (epoch +1,num_loo, running_loss/len(train_index)))
            
            """
            Aggregate global model using 'mean'
            """
            ### This will take simple mean of the weights of models ###
            global_dict = global_model.state_dict()
            for k in global_dict.keys():
                global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
            

            """
            This function test the global model on test data and returns test loss and test accuracy 
            """
            global_model.load_state_dict(global_dict)
            global_model.eval()
            #Evaluation of model
            running_loss = 0.0  
            running_error = 0.0 

            with torch.no_grad():
                for i in range(len(test_index)):
                    loader = dataloaders_list[test_index[i]] 
                    for j, data in enumerate(loader,1) :  
                        images, labels = data
                        if use_cuda and torch.cuda.is_available():
                            images = images.cuda() 
                            labels = labels.cuda()

                        outputs = global_model(images.float())
                    running_loss += loss_cnn(outputs, labels.float()).item() 
                    running_error += mse(labels.float(),outputs) 
            
            
            epoch_loss = running_loss / len(test_index)
            epoch_error = running_error / len(test_index)
            val_history.append(epoch_loss)
                        
            print('Val Epoch[%d] LOO[%d]  loss: %.8f Valence MSE: %.8f Arousal MSE: %.8f'  %
                  (epoch +1, num_loo, epoch_loss, epoch_error[0],epoch_error[1]))
            
            # deep copy the model
            if epoch_loss < best_loss and epoch_error[0] < valence_best_error and epoch_error[1] < arousal_best_error:
                best_loss = epoch_loss
                valence_best_error = epoch_error[0]
                arousal_best_error = epoch_error[1]
                best_model_wts = copy.deepcopy(global_model.state_dict())
                #torch.save(model.state_dict(),file)            

            global_model.load_state_dict(best_model_wts)
            
        num_loo +=1    
        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:8f}'.format(best_loss))
        print('Best Valence error: {:8f}'.format(valence_best_error))
        print('Best Arousal error: {:8f}'.format(arousal_best_error))



        # load best model weights test_index[-1]
        torch.save(global_model.state_dict(),file)

        #Evaluate model performance
        actual = []
        pred = []
        global_model.eval()
        with torch.no_grad():
            for i in range(len(test_index)):
                loader = dataloaders_list[test_index[i]] 
                for data in loader:
                    images, labels = data
                    if use_cuda and torch.cuda.is_available():
                        images = images.cuda()   

                    outputs = global_model(images.float())

                    actual.extend(labels.numpy())
                    pred.extend(outputs.detach().cpu().numpy())

        print(np.array(actual).shape)

        print('Valence MSE: %.8f Arousal MSE: %.8f ' % (mse1(np.array(actual)[:,0],np.array(pred)[:,0]),mse1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence MAE: %.8f Arousal MAE: %.8f ' % (merror1(np.array(actual)[:,0],np.array(pred)[:,0]),merror1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence RMSE: %.8f Arousal RMSE: %.8f' % (rmse1(np.array(actual)[:,0],np.array(pred)[:,0]),rmse1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence CCC: %.8f Arousal CCC: %.8f' % (ccc(np.array(actual)[:,0],np.array(pred)[:,0]),ccc(np.array(actual)[:,1],np.array(pred)[:,1])))
        print_dropped(mdl,input_features, seq_len,hidden_size,num_layers,num_loo,
                      mse1(np.array(actual)[:,0],np.array(pred)[:,0]),mse1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     merror1(np.array(actual)[:,0],np.array(pred)[:,0]),merror1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     rmse1(np.array(actual)[:,0],np.array(pred)[:,0]),rmse1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     ccc(np.array(actual)[:,0],np.array(pred)[:,0]),ccc(np.array(actual)[:,1],np.array(pred)[:,1]))
        list_models.append(global_model)
        print(train_history)
        print(val_history)



    return list_models



mdls = train_network(list_dataloaders, df_part,  num_epochs, file,input_features, hidden_size,num_layers,output_dim,lr,seq_len)

