# UNI: jjl2245, me2680
"""The main demo script for our project which runs our best model on a randomized subset of the test data"""

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

# Load our best saved model
LOAD_PATH = '../../saved_models/tdnn_12epoch_allLang_alltraindata.pth'
print('Loading our best model')
net = Net()
net.to(device)
net.load_state_dict(torch.load(LOAD_PATH, map_location=device))

# Get test data
print("Getting test data")
root_path = '../../features/mfcc_pitch/'
langs = ['EN', 'ES', 'FR', 'KR', 'MD', 'RS', 'TR', 'BN', 'JV'] # All features for all 9 languages (7 in-set and 2 out-of-set)
file_name = 'raw_mfcc_pitch_test.1.ark'

test = []
for i,lang in enumerate(langs,0):
    for key, numpy_array in kaldiio.load_ark(root_path + lang + '/' + file_name):
        inputs = torch.from_numpy(np.expand_dims(numpy_array, axis=0))
        inputs.to(device)
        labels = torch.from_numpy(np.array([i]))
        if lang == 'BN' or lang == 'JV': # Out-of-set languages will have label -1
            labels = torch.from_numpy(np.array([-1]))
        labels.to(device)
        test.append((inputs,labels))
print()
random.shuffle(test) # randomize order of test data
test = test[:300] # take a random 300 size sample of the test data

# Perform the test
index_map = {0:'English (IN-SET)', 1:'Spanish (IN-SET)', 2:'French (IN-SET)', 3:'Korean (IN-SET)', 4:'Mandarin (IN-SET)', 5:'Russian (IN-SET)', 6:'Turkish (IN-SET)', -1:'OUT-OF-SET'}
threshold = 0.8
correct_per_set = {'in': 0, 'out': 0}
total_per_set = {'in': 0, 'out': 0}
correct = 0
total = 0
with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
    for i, data in enumerate(test,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        outputs = torch.mean(outputs, 1)
    
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)

        # apply threshold function
        max_energy = torch.max(outputs.data)
        if max_energy < threshold:
            predicted = torch.from_numpy(np.array([-1])).to(device)

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

        print('True label:\t\t', index_map[labels.item()])
        print('Predicted language:\t', index_map[predicted.item()])
        print()

    print(f'Overall accuracy on the test data: {correct / total}')
    print('Accuracy on in-set test data:', correct_per_set['in'] / total_per_set['in'])
    print('Accuracy on out-of-set test data:', correct_per_set['out'] / total_per_set['out'])
    print()




