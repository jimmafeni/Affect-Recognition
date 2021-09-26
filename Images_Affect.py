
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
dir_video = 'folder containing images of participants'
arousal_file = 'folder for arousal groundtruth of each image'
valence_file = 'folder for valence groundtruth of each image'
image_names = 'file containing names of images'
part_file = 'file containing names of participants'
log_file = 'log file'


# Face extractor function
mtcnn = MTCNN(image_size=224,min_face_size=10, thresholds=[0.5, 0.5, 0.5])

#load participant file

df_part = pd.read_csv(part_file)


# Results output function



def print_dropped(md,feat,sql,hs,nl,part,mse,mse2,mae,mae2,
                  rmse, rmse2,ccc,ccc2,file=log_file ):    
    pd.DataFrame({'a':[md],'n':[feat],'b':[sql],'c':[hs],'d':[nl],
                  'e':[part],'f':[mse],'g':[mse2],'h':[mae],'i':[mae2],
                  'j':[rmse], 'k':[rmse2],'l':[ccc],'m':[ccc2]}).to_csv(file, mode='a', header=False)  
    print('Added to output file')    


# Performance evaluation metrics


def merror2(true,pred):
    d=np.abs(pred.detach().cpu().numpy()-true.detach().cpu().numpy())
    return d



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




def aggre_annot(annot):
    
    return round(annot.iloc[:,1:].mean(axis=1),4)   


# Dataloader function


# Normalisation
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#Custom dataset loader creation
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


        return image, label

#extract each participants data and store in a dataloader: X->features Y-> 2dimensional(valence and arousal)
def extract_data(df_part, dir_video, image_names , arousal_file, valence_file, batch_size):
    #list to store dataloaders
    list_dataloaders=[]
    
    for i in range(len(df_part)):
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
        dataloaders = torch.utils.data.DataLoader(train_data, batch_size = batch_size,shuffle = False,drop_last = True, num_workers=4)      
        #add dataloader to the list of dataloaders
        list_dataloaders.append(dataloaders)
    
    return list_dataloaders
    


# Set model parameters

# Sequence length
seq_len = 16

# batch_size
batch_size = 16

# number of features
input_features = 512

# number of hidden units
hidden_size = 128

#number lstm layers
num_layers = 2

#output dimension
output_dim = 2


#learning rate
lr = 0.0001

#number of epochs
num_epochs = 30

#model name
mdl = 'CNN_LSTMsizetest'

#model file
file = "CNN_BiLSTM_raw.pth"

#MODELS

#CNN model to extract spatial features

def CNN():
    
    preNet = models.resnet18(pretrained=True)
    
    #for i, child in enumerate(preNet.children(),0):
        #print(child,i)
        #if i < 11:
            #for param in child.parameters():
                #param.requires_grad = False

    preNet.fc = nn.Sequential()
    #num_ftrs = preNet.fc.in_features
    #preNet.fc = nn.Linear(num_ftrs, num_outputs)

    return preNet
	
#RNN MODELS coupled with CNNs for sequence learning

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
        
        self.out1 = nn.Linear(hidden_size*2*seq_len, 1000)  
        
        self.out2 = nn.Linear(1000, output_dim) 
        

        
    def forward(self, input):
        


        input1 = self.cnn(input)


               
        output, hidden = self.lstm(input1.view(input1.shape[0],-1,input_features))
        
        
        #pred = self.out1(output.view(-1,hidden_size*2*seq_len))     

             
        pred = self.out2(self.out1(output.view(-1,hidden_size*2*seq_len)))

        return pred[-1]




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


#Create loader list
list_dataloaders = extract_data(df_part, dir_video, image_names, arousal_file, valence_file, batch_size)


# Initialise model



def initialise_model(input_features, hidden_size,num_layers,output_dim,lr,seq_len):
    
    model = BiLSTM(input_features, hidden_size,num_layers,output_dim,seq_len)

    
    use_cuda = True

    if use_cuda and torch.cuda.is_available():
        model.cuda()


    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, loss, optimizer, exp_lr_scheduler


# Training function




def train_network(dataloaders_list, list_part, epochs, file, input_features, hidden_size,num_layers,output_dim,lr,seq_len):
    #use cuda
    use_cuda = True

    #Leave-one-out cross validation
    k = KFold(8)
    num_folds = 1
    list_models = []
    
    #leave-one-out cross-validation
    for train_index, test_index in k.split(dataloaders_list):
        
        model, loss_cnn, optimizer,scheduler = initialise_model(input_features, hidden_size,num_layers,output_dim,lr,seq_len)

     
        since = time.time()       

        train_history = []
        val_history = []

        #Validate model and store best parameters
        best_model_wts = copy.deepcopy(model.state_dict())
        valence_best_error = 50.0
        arousal_best_error = 50.0
        best_loss = 50.0

        for epoch in range(epochs): 

            # Set model to training mode
            model.train()  

            running_loss = 0.0  
            running_error = 0.0 


            for i in range(len(train_index)):
                loader = dataloaders_list[train_index[i]] 

                for j, data in enumerate(loader, 0):
                    images, labels = data

                    if use_cuda and torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()


                    # zero the parameter gradients
                    optimizer.zero_grad() 

                    with torch.set_grad_enabled(True):

                        outputs = model(images.float())

                        loss = loss_cnn(outputs, torch.mean(labels.float(),0))

                        # backward + optimize                       
                        loss.backward()
                        optimizer.step()


            # statistics
                running_loss += loss.item() 

                running_error += merror2(torch.mean(labels.float(),0),outputs)



            scheduler.step()
            epoch_loss = running_loss / len(train_index)
            epoch_error = running_error / len(train_index)
            train_history.append(epoch_loss)


            print('LOO[%d] Train Epoch [%d] loss: %.8f Valence MSE: %.8f Arousal MSE: %.8f'  % 
                  (num_folds, epoch + 1, epoch_loss, epoch_error[0],epoch_error[1]))
            
            #Evaluation of model
            running_loss = 0.0  
            running_error = 0.0 
            
            model.eval()            

            with torch.no_grad():
                for i in range(len(test_index)):
                    loader = dataloaders_list[test_index[i]] 
                    for j, data in enumerate(loader,1) :  
                        images, labels = data
                        if use_cuda and torch.cuda.is_available():
                            images = images.cuda() 
                            labels = labels.cuda()

                        outputs = model(images.float())
                    running_loss += loss_cnn(outputs, torch.mean(labels.float(),0)).item() 
                    running_error += merror2(torch.mean(labels.float(),0),outputs)  

            
            
            epoch_loss = running_loss / len(test_index)
            epoch_error = running_error / len(test_index)
            val_history.append(epoch_loss)
                        
            print('LOO[%d] Val Epoch[%d] loss: %.8f Valence MSE: %.8f Arousal MSE: %.8f'  %
                  (num_folds, epoch + 1, epoch_loss, epoch_error[0],epoch_error[1]))
            
            
            
            # deep copy the model
            if epoch_loss < best_loss and epoch_error[0] < valence_best_error and epoch_error[1] < arousal_best_error:
                best_loss = epoch_loss
                valence_best_error = epoch_error[0]
                arousal_best_error = epoch_error[1]
                best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(),file)            
                
        print()
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:8f}'.format(best_loss))
        print('Best Valence error: {:8f}'.format(valence_best_error))
        print('Best Arousal error: {:8f}'.format(arousal_best_error))
        
        
        
        # load best model weights test_index[-1]
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(),file)
        
        #Evaluate model performance
        actual = []
        pred = []
        with torch.no_grad():
            for i in range(len(test_index)):
                loader = dataloaders_list[test_index[i]] 
                for data in loader:
                    model.eval()
                    images, labels = data
                    if use_cuda and torch.cuda.is_available():
                        images = images.cuda()   

                    outputs = model(images.float())

                    actual.append(torch.mean(labels.float(),0).numpy())
                    pred.append(outputs.detach().cpu().numpy())

        print(np.array(actual).shape)
        
        print('Valence MSE: %.8f Arousal MSE: %.8f ' % (mse1(np.array(actual)[:,0],np.array(pred)[:,0]),mse1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence MAE: %.8f Arousal MAE: %.8f ' % (merror1(np.array(actual)[:,0],np.array(pred)[:,0]),merror1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence RMSE: %.8f Arousal RMSE: %.8f' % (rmse1(np.array(actual)[:,0],np.array(pred)[:,0]),rmse1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence CCC: %.8f Arousal CCC: %.8f' % (ccc(np.array(actual)[:,0],np.array(pred)[:,0]),ccc(np.array(actual)[:,1],np.array(pred)[:,1])))
        print_dropped(mdl,input_features, seq_len,hidden_size,num_layers,num_folds,
                      mse1(np.array(actual)[:,0],np.array(pred)[:,0]),mse1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     merror1(np.array(actual)[:,0],np.array(pred)[:,0]),merror1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     rmse1(np.array(actual)[:,0],np.array(pred)[:,0]),rmse1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     ccc(np.array(actual)[:,0],np.array(pred)[:,0]),ccc(np.array(actual)[:,1],np.array(pred)[:,1]))
        
        list_models.append(model)
        print(train_history)
        print(val_history)

        num_folds +=1

    return list_models



# In[648]:


mdls = train_network(list_dataloaders, df_part,  num_epochs, file,input_features, hidden_size,num_layers,output_dim,lr,seq_len)


