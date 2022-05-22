# UNI: jjl2245, me2680
"""This script implements the TDNN model and runs the training and testing experiments for various thresholds"""

import kaldiio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random


# Use GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class TDNN(nn.Module):
    '''Implementation of TDNN (Time Delay Neural Network) layers'''

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1, batch_norm=True, dropout_p=0.0):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        output: size (batch, new_seq_len, output_features)
        '''
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(self.input_dim, d)
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
                        x, 
                        (self.context_size, self.input_dim), 
                        stride=(1,self.input_dim), 
                        dilation=(self.dilation,1)
                    )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1,2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        
        if self.dropout_p:
            x = self.drop(x)
        if self.batch_norm:
            x = x.transpose(1,2)
            x = self.bn(x)
            x = x.transpose(1,2)

        return x

class Net(nn.Module):
    '''This class constructs the TDNN'''

    def __init__(self):
        super().__init__()
        self.layer1 = TDNN(input_dim=16, output_dim=512, context_size=5, dilation=1)
        self.layer2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.layer3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.layer4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.layer5 = TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.layer6 = TDNN(input_dim=1500, output_dim=7, context_size=1, dilation=1)
        self.layer7 = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


# Initialize the TDNN, loss, and optimizer
net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss() # a common loss function for multi-class classification problems
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # a common optimizer for multi-class classification problems




# Get training data
print("Getting training data")
root_path = '../../features/mfcc_pitch/'
langs = ['EN', 'ES', 'FR', 'KR', 'MD', 'RS', 'TR'] # Get all features from the 7 in-set languages
file_name = 'raw_mfcc_pitch_train.1.ark'

train = []
for i,lang in enumerate(langs,0): # for each language
    for key, numpy_array in kaldiio.load_ark(root_path + lang + '/' + file_name): # read the ark file
        inputs = torch.from_numpy(np.expand_dims(numpy_array, axis=0))
        inputs.to(device)
        labels = torch.from_numpy(np.array([i])) # label the training data with index as the label
        labels.to(device)
        train.append((inputs,labels))

print("Shuffling training data")
random.shuffle(train) # shuffling is important for training
print("training data length: ", len(train))
print()




# Train the TDNN network
do_training = True # whether to train or to load a saved model
SAVE_PATH = '../../saved_models/tdnn.pth'
LOAD_PATH = '../../saved_models/tdnn_12epoch_allLang_alltraindata.pth'

if do_training:
    print('Started Training')

    for epoch in range(12):  # number of epochs
        random.shuffle(train) # shuffle data every epoch
        print('\tepoch: ', str(epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs) # pass inputs thorugh the model
            outputs = torch.mean(outputs, 1) #average over all time slices
            loss = criterion(outputs, labels) # compute loss
            loss.backward() # compute gradients
            optimizer.step() # SGD step
            
            # print stats for running loss
            running_loss += loss.item()
            report_interval = len(train) // 10 # 10 running losses get reported each epoch
            if i % report_interval == report_interval-1:
                print(f'\t\t[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('Finished Training')
    torch.save(net.state_dict(), SAVE_PATH) # Save the model

else:
    # Skip training and load a saved model
    print('Skip training')
    print('Loading the model')
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(LOAD_PATH, map_location=device))
    print('Finished loading')
print()




# Get test data
print("Getting testing data")
root_path = '../../features/mfcc_pitch/'
langs = ['EN', 'ES', 'FR', 'KR', 'MD', 'RS', 'TR', 'BN', 'JV'] # All features for all 9 languages (7 in-set and 2 out-of-set)
file_name = 'raw_mfcc_pitch_test.1.ark'

test = []
for i,lang in enumerate(langs,0): # for each language
    for key, numpy_array in kaldiio.load_ark(root_path + lang + '/' + file_name): # read the ark file
        inputs = torch.from_numpy(np.expand_dims(numpy_array, axis=0))
        inputs.to(device)
        labels = torch.from_numpy(np.array([i]))
        if lang == 'BN' or lang == 'JV': # Out-of-set languages will have label -1
            labels = torch.from_numpy(np.array([-1]))
        labels.to(device)
        test.append((inputs,labels))
print("testing data length: ", len(test))
print()




print("Testing...")
for threshold in [0,0.5,0.6,0.7,0.8,0.9]: # Testing different thresholds
    correct_per_set = {'in': 0, 'out': 0} # counts correct for in-set and out-of-set
    total_per_set = {'in': 0, 'out': 0} # counts the total test data points
    correct = 0 # correct (used for overall accuracy)
    total = 0 # total (used for overall accuracy)
    with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
        for i, data in enumerate(test,0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs) # run inputs through the model to get outputs
            outputs = torch.mean(outputs, 1) # average the output of the network over time slices
        
            _, predicted = torch.max(outputs.data, 1) # the class with the highest energy is what we choose as prediction

            # apply threshold function
            max_energy = torch.max(outputs.data)
            if max_energy < threshold:
                predicted = torch.from_numpy(np.array([-1])).to(device)

            # Compute overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute accuracy for inset and outofset separately
            if labels == torch.from_numpy(np.array([-1])).to(device):
                total_per_set['out'] += 1
                if predicted == labels:
                    correct_per_set['out'] += 1
            else:
                total_per_set['in'] += 1
                if predicted == labels:
                    correct_per_set['in'] += 1

     

    print(f'Threshold: {threshold}')
    print(f'Overall accuracy on the test data: {correct / total}')
    print('Accuracy on in-set test data:', correct_per_set['in'] / total_per_set['in'])
    print('Accuracy on out-of-set test data:', correct_per_set['out'] / total_per_set['out'])
    print()






'''
RESULTS

Basic 7-language identification (no out of set langs):
    8 epoch, all 7 langs, all training data:    0.9400733795352629
    12 epoch, all 7 langs, all training data:   0.9518956379942927
    20 epoch, all 7 langs, all training data:   0.9445576844679984

Picked best model (12 epoch) and added out-of-set test-data and thresholds to it:
    Accuracy of the network on the test data with threshold 0.1:    0.5702772011234583
    Accuracy of the network on the test data with threshold 0.15:   0.5702772011234583
    Accuracy of the network on the test data with threshold 0.2:    0.5703993161558187
    Accuracy of the network on the test data with threshold 0.25:   0.5728416168030285
    Accuracy of the network on the test data with threshold 0.3:    0.5822444742947857
    Accuracy of the network on the test data with threshold 0.35:   0.5986078886310905
    Accuracy of the network on the test data with threshold 0.4:    0.621443399682501
    Accuracy of the network on the test data with threshold 0.45:   0.6468433264134815
    Accuracy of the network on the test data with threshold 0.5:    0.6750518988887532
    Accuracy of the network on the test data with threshold 0.55:   0.6997191354255708
    Accuracy of the network on the test data with threshold 0.6:    0.7290267431920869
    Accuracy of the network on the test data with threshold 0.65:   0.7552814751495909
    Accuracy of the network on the test data with threshold 0.7:    0.7851996580779094
    Accuracy of the network on the test data with threshold 0.75:   0.8163389913298327
    Accuracy of the network on the test data with threshold 0.8:    0.833068750763219
    Accuracy of the network on the test data with threshold 0.85:   0.7238979118329466
    Accuracy of the network on the test data with threshold 0.9:    0.542679203809989

Inset and outofset accuracies with just the best thresholds:
    Threshold: 0.6
        Overall accuracy on the test data: 0.7290267431920869
        Accuracy on in-set test data: 0.930900937627395
        Accuracy on out-of-set test data: 0.4273530307645446

    Threshold: 0.65
        Overall accuracy on the test data: 0.7552814751495909
        Accuracy on in-set test data: 0.9237668161434978
        Accuracy on out-of-set test data: 0.5035028936947914

    Threshold: 0.7
        Overall accuracy on the test data: 0.7851996580779094
        Accuracy on in-set test data: 0.9135752140236445
        Accuracy on out-of-set test data: 0.5933597319524825

    Threshold: 0.75
        Overall accuracy on the test data: 0.8163389913298327
        Accuracy on in-set test data: 0.8931920097839381
        Accuracy on out-of-set test data: 0.7014925373134329

    Threshold: 0.7625
        Overall accuracy on the test data: 0.8225668579802173
        Accuracy on in-set test data: 0.8864655523848349
        Accuracy on out-of-set test data: 0.7270788912579957

    Threshold: 0.775
        Overall accuracy on the test data: 0.8267187690804738
        Accuracy on in-set test data: 0.877293110476967
        Accuracy on out-of-set test data: 0.7511422479439537

    Threshold: 0.7875
        Overall accuracy on the test data: 0.8318476004396141
        Accuracy on in-set test data: 0.8673053403995108
        Accuracy on out-of-set test data: 0.7788607980505635

    Threshold: 0.8
        Overall accuracy on the test data: 0.833068750763219
        Accuracy on in-set test data: 0.852221769262128
        Accuracy on out-of-set test data: 0.8044471519951264

    Threshold: 0.8125
        Overall accuracy on the test data: 0.8312370252778117
        Accuracy on in-set test data: 0.8291887484712597
        Accuracy on out-of-set test data: 0.8342978982637831

    Threshold: 0.825
        Overall accuracy on the test data: 0.8160947612651117
        Accuracy on in-set test data: 0.7867916836526702
        Accuracy on out-of-set test data: 0.8598842522083461
'''
