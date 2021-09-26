
# coding: utf-8

# In[191]:


import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import torch.utils.data.dataloader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification,classification_report,confusion_matrix,accuracy_score
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torchsummary import summary
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import LeaveOneOut,KFold
import time
import os
import copy
from sklearn import preprocessing


# folders containing extracted AUs and groundtruth
dir_features = 'folder containing AUs features of images for each participant'
arousal_file = 'arousal folder containing arousal groudtruth of images for each participant'
valence_file = 'valence folder containing valence groudtruth of images for each participant'
file_part = 'participants definition file'
log_file = 'log file'





#Load participant file

df_part = pd.read_csv(file_part)


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


# In[636]:


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
def ccc(y_true, y_pred,sample_weight=None,multioutput='uniform_average'):
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
    
    return round(annot.iloc[:,1:].mean(axis=1),2)   


# Dataloader function



def create_dataloaders(x,y, seq_len):
    data = torch.utils.data.TensorDataset(torch.tensor(x.values), torch.tensor(y.values))
    dataloaders = torch.utils.data.DataLoader(data, batch_size = seq_len, shuffle = False,drop_last = True, num_workers=4)
    
    return dataloaders
    


#extract each participants data and store in a dataloader: X->features Y-> 2dimensional(valence and arousal)
def extract_data(df_part, dir_features, arousal_file, valence_file, seq_len):
    #list to store dataloaders
    list_dataloaders=[]
    
    for i in range(len(df_part)):
        partFile = df_part.Participants[i]
        #features csv
        df_features = pd.read_csv(dir_features + partFile)
        df_features = df_features.iloc[:,1:]
        #save column names
        names = df_features.columns

        # Create the Scaler object
        scaler = preprocessing.StandardScaler()

        # Fit your data on the scaler object
        df_features  = scaler.fit_transform(df_features)
        df_features  = pd.DataFrame(df_features , columns=names)        
        
        #valence csv
        df_valence = pd.read_csv(valence_file + partFile,sep=';')
        #drop first row of annotation because row missing in features csv
        df_valence.drop(df_valence.index[0],inplace=True)
        #arousal csv
        df_arousal = pd.read_csv(arousal_file + partFile,sep=';')
        df_arousal.drop(df_arousal.index[0],inplace=True)
        #get aggregate of the annotations e.g. average
        df_valence = aggre_annot(df_valence)
        df_arousal = aggre_annot(df_arousal)
        #merge valence and arousal -> Y
        Y = pd.DataFrame({'valence':df_valence,'arousal':df_arousal})
        #creat dataloaders
        dataloaders = create_dataloaders(df_features, Y,seq_len)
        #add dataloader to the list of dataloaders
        list_dataloaders.append(dataloaders)
    
    return list_dataloaders

    


# Set model parameters


# Sequence Length
seq_len = 600

# number of features
input_features = 40

# number of hidden units
hidden_size = 512

#number lstm layers
num_layers = 6

#output dimension
output_dim = 2

#learning rate
lr = 0.0001

#number of epochs
num_epochs = 70

#model name
mdl = 'BiBGRUlasttest'

#model file
file = "BiGRU.pth"


#MODELS

#BiLSTM classification model
class BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim,seq_len):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers=num_layers, bidirectional=True)
        
        self.out1 = nn.Linear(hidden_size*2*seq_len, 1000)  
        
        self.out2 = nn.Linear(1000, output_dim) 
        

        
    def forward(self, input):
               
        output, hidden = self.lstm(input.view(input.shape[0],-1,input_features))  
             
        pred = self.out2(self.out1(output.view(-1,seq_len*hidden_size*2)))        


        return pred[-1]


#LSTM classification model
class LSTM_model(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim):
        super(LSTM_model, self).__init__()
        
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers=num_layers, bidirectional=False)
        
        self.out1 = nn.Linear(hidden_size, 10)  
        
        self.out2 = nn.Linear(10, output_dim) 
        

        
    def forward(self, input):
               
        output, hidden = self.lstm(input.view(input.shape[0],-1,input_features))       
             
        pred = self.out2(self.out1(output.view(-1,hidden_size)))

        return pred

#RNN classification model
class RNN_model(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim):
        super(RNN_model, self).__init__()
        
        self.rnn = nn.RNN(input_features, hidden_size, num_layers=num_layers)
        
        self.out1 = nn.Linear(hidden_size, 10)  
        
        self.out2 = nn.Linear(10, output_dim) 
        

        
    def forward(self, input):
               
        output, hidden = self.rnn(input.view(input.shape[0],-1,input_features))       
             
        pred = self.out2(self.out1(output.view(-1,hidden_size)))

        return pred

#BiGRU classification model
class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim,seq_len):
        super(BiGRU, self).__init__()
        
        self.gru = nn.GRU(input_features, hidden_size, num_layers=num_layers, bidirectional=True)
        
        self.out1 = nn.Linear(hidden_size*2*seq_len, 1000) 

        #self.out2 = nn.Linear(50, 50) 
        
        self.out3 = nn.Linear(1000, output_dim) 
        

        
    def forward(self, input):
               
        output, hidden = self.gru(input.view(input.shape[0],-1,input_features)) 

        #pred = self.out1(output.view(-1,hidden_size*2))

        pred = self.out3(self.out1(output.view(-1,hidden_size*2*seq_len)))   
             
        #pred = self.out3(self.out2(self.out1(output.view(-1,hidden_size*2))))

        return pred[-1]

#Residual BiLSTM classification model
class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 
    
class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()        
        
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        
        return x # (batch, channel, feature, time)
    
class ReBiLSTM(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim):
        super(ReBiLSTM, self).__init__()
        
        self.RNN_model = nn.Sequential(*[
            ResidualCNN(1, 1, kernel=3, stride=1, dropout=0.1, n_feats = input_features) 
            for _ in range(2)
        ])
        
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers=num_layers, bidirectional=True)
        
        self.out1 = nn.Linear(hidden_size*2, 10)  
        
        self.out2 = nn.Linear(10, output_dim) 
        

        
    def forward(self, input):
        resi_feat = self.RNN_model(input.float().view(1,1,input_features,-1))
               
        output, hidden = self.lstm(resi_feat.view(input.shape[0],-1,input_features))       
             
        pred = self.out2(self.out1(output.view(-1,hidden_size*2)))

        return pred

class ReBiGRU(nn.Module):
    def __init__(self, input_features, hidden_size,num_layers,output_dim):
        super(ReBiGRU, self).__init__()
        
        self.RNN_model = nn.Sequential(*[
            ResidualCNN(1, 1, kernel=3, stride=1, dropout=0.1, n_feats = input_features) 
            for _ in range(2)
        ])
        
        self.gru = nn.GRU(input_features, hidden_size, num_layers=num_layers, bidirectional=True)
        
        self.out1 = nn.Linear(hidden_size*2, 10)  
        
        self.out2 = nn.Linear(10, output_dim) 
        

        
    def forward(self, input):
        resi_feat = self.RNN_model(input.float().view(1,1,input_features,-1))
               
        output, hidden = self.gru(resi_feat.view(input.shape[0],-1,input_features))       
             
        pred = self.out2(self.out1(output.view(-1,hidden_size*2)))

        return pred



# Create dataloaders and model



list_dataloaders = extract_data(df_part, dir_features, arousal_file, valence_file, seq_len)


# Initialise model



def initialise_model(input_features, hidden_size,num_layers,output_dim,lr,seq_len):
    model = BiGRU(input_features, hidden_size,num_layers,output_dim,seq_len)
    
    use_cuda = True

    if use_cuda and torch.cuda.is_available():
        model.cuda()


    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    return model, loss, optimizer


# Training function



def train_network(dataloaders_list, list_part, epochs, file, input_features, hidden_size,num_layers,output_dim,lr,seq_len):
    #use cuda
    use_cuda = True

    #Cross validation
    loo = KFold(8)
    num_loo = 1
    list_models = []
    
    #leave-one-out cross-validation
    for train_index, test_index in loo.split(dataloaders_list):
        
        model, loss_cnn, optimizer = initialise_model(input_features, hidden_size,num_layers,output_dim,lr,seq_len)        
        
        since = time.time()       

        train_history = []

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
            len_data = 0.0

            for i in range(len(train_index)):
                loader = dataloaders_list[train_index[i]] 
                len_data += len(loader.dataset)
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
#                         loss2 = loss_cnn(outputs, labels.float())
#                         loss = loss1 + loss2

                        #preds = torch.round(torch.sigmoid(outputs))

                        # backward + optimize only if in training phase                        
                        loss.backward()
                        optimizer.step()


                    # statistics
                    running_loss += loss.item() * images.size(0)
                    running_error += merror2(torch.mean(labels.float(),0),outputs) * images.size(0)



            epoch_loss = running_loss / len_data
            epoch_error = running_error / len_data


            print('LOO[%s] Train Epoch [%d] loss: %.8f Valence MAE: %.8f Arousal MAE: %.8f'  % 
                  (list_part.Participants[test_index[-1]], epoch + 1, epoch_loss, epoch_error[0],epoch_error[1]))
            
            #Evaluation of model
            running_loss = 0.0  
            running_error = 0.0 
            
            model.eval()            
            
            with torch.no_grad():
				for i in range(len(test_index)):
					for data in dataloaders_list[test_index[i]] :  
						images, labels = data
						if use_cuda and torch.cuda.is_available():
							images = images.cuda()
							labels = labels.cuda()               

						outputs = model(images.float())
						running_loss += loss_cnn(outputs, torch.mean(labels.float(),0)).item() 
						running_error += merror2(torch.mean(labels.float(),0),outputs) 
				
            
            epoch_loss = running_loss / len(dataloaders_list[test_index[-1]])
            epoch_error = running_error / len(dataloaders_list[test_index[-1]])
                        
            print('LOO[%s] Val Epoch[%d] loss: %.8f Valence MAE: %.8f Arousal MAE: %.8f'  %
                  (list_part.Participants[test_index[-1]], epoch + 1, epoch_loss, epoch_error[0],epoch_error[1]))
            
            
            
            # deep copy the model
            if epoch_loss < best_loss and epoch_error[0] < valence_best_error and epoch_error[1] < arousal_best_error:
                best_loss = epoch_loss
                valence_best_error = epoch_error[0]
                arousal_best_error = epoch_error[1]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),file)            
                train_history.append(epoch_error)
        print()
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best loss: {:8f}'.format(best_loss))
        print('Best Valence error: {:8f}'.format(valence_best_error))
        print('Best Arousal error: {:8f}'.format(arousal_best_error))
        
        num_loo +=1
        
        # load best model weights
        model.load_state_dict(best_model_wts)
        
        
        #Evaluate model performance
        actual = []
        pred = []
        with torch.no_grad():
			for i in range(len(test_index)):
				for data in dataloaders_list[test_index[i]] :
					model.eval()
					images, labels = data
					if use_cuda and torch.cuda.is_available():
						images = images.cuda()   

					outputs = model(images.float())
					#predicted = torch.round(torch.sigmoid(outputs))

					actual.append(torch.mean(labels.float(),0).numpy())
					pred.append(outputs.detach().cpu().numpy())
        #print(np.array(actual))
        #print(np.array(pred))

        print('Valence MSE: %.8f Arousal MSE: %.8f ' % (mse1(np.array(actual)[:,0],np.array(pred)[:,0]),mse1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence MAE: %.8f Arousal MAE: %.8f ' % (merror1(np.array(actual)[:,0],np.array(pred)[:,0]),merror1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence RMSE: %.8f Arousal RMSE: %.8f' % (rmse1(np.array(actual)[:,0],np.array(pred)[:,0]),rmse1(np.array(actual)[:,1],np.array(pred)[:,1])))
        print('Valence CCC: %.8f Arousal CCC: %.8f' % (ccc(np.array(actual)[:,0],np.array(pred)[:,0]),ccc(np.array(actual)[:,1],np.array(pred)[:,1])))

        #print in output file
        print_dropped(mdl, input_features, seq_len,hidden_size,num_layers,list_part.Participants[test_index[-1]],
                      mse1(np.array(actual)[:,0],np.array(pred)[:,0]),mse1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     merror1(np.array(actual)[:,0],np.array(pred)[:,0]),merror1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     rmse1(np.array(actual)[:,0],np.array(pred)[:,0]),rmse1(np.array(actual)[:,1],np.array(pred)[:,1]),
                     ccc(np.array(actual)[:,0],np.array(pred)[:,0]),ccc(np.array(actual)[:,1],np.array(pred)[:,1]))
                
        list_models.append(model)

    return list_models


# In[648]:


models = train_network(list_dataloaders, df_part,  num_epochs, file,input_features, hidden_size,num_layers,output_dim,lr,seq_len)


