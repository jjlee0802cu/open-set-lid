# UNI: jjl2245, me2680
"""This script implements the CRNN with attention model and runs the training and testing experiments for various thresholds"""

import kaldiio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
from torch.autograd import Variable

# Use GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


class BidirectionalLSTM(nn.Module):
    """Implementation of BiLSTM"""
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class AttentionCell(nn.Module):
    "Implementation of Attention Cell"
    def __init__(self, input_size, hidden_size):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.processed_batches = 0

    def forward(self, prev_hidden, feats):
        self.processed_batches = self.processed_batches + 1
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(torch.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        alpha = F.softmax(emition, dim=1) # nB * nT

        # if self.processed_batches % 10000 == 0:
        #     print('emition ', list(emition.data[0]))
        #     print('alpha ', list(alpha.data[0]))

        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha

class Attention(nn.Module):
    """Implementation of Attention"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.processed_batches = 0

    def forward(self, feats, text_length):
        text_length = torch.IntTensor([1]).to(device)
        text_length = text_length.repeat(feats.size(1)).to(device)

        self.processed_batches = self.processed_batches + 1
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert(input_size == nC)
        assert(nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()

        output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
        hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
        max_locs = torch.zeros(num_steps, nB)
        max_vals = torch.zeros(num_steps, nB)
        for i in range(num_steps):
            hidden, alpha = self.attention_cell(hidden, feats)
            output_hiddens[i] = hidden
            if self.processed_batches % 500 == 0:
                max_val, max_loc = alpha.data.max(1)
                max_locs[i] = max_loc.cpu()
                max_vals[i] = max_val.cpu()
        # if self.processed_batches % 500 == 0:
        #     print('max_locs', list(max_locs[0:text_length.data[0],0]))
        #     print('max_vals', list(max_vals[0:text_length.data[0],0]))
        new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
        b = 0
        start = 0
        for length in text_length.data:
            new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
            start = start + length
            b = b + 1
        probs = self.generator(new_hiddens)
        return probs

class Net(nn.Module):
    '''This class constructs the CRNN'''
    def __init__(self):
        super().__init__() 

        # CNN
        self.conv1 = nn.Conv2d(1, 1, 2)
        self.conv2 = nn.Conv2d(1, 1, 2)

        # RNN + Attention
        self.layer1 = BidirectionalLSTM(20, 512, 512)
        self.layer2 = BidirectionalLSTM(512, 512, 512)
        self.attention = Attention(512, 512, 7)
        self.layer3 = nn.Softmax(dim=1)
    

    def forward(self, x):
        # Format input for CNN
        y = x.reshape((x.shape[0], 4,4)) # ts x 4 x 4
        y = np.expand_dims(y, 1) # ts x 1 x 4 x 4
        y = torch.from_numpy(y).to(device)

        # Pass through CNN
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = torch.flatten(y, 1) # ts x 4

        # Format input for RNN
        x = torch.from_numpy(x).to(device)
        x = torch.cat((x,y), -1) # Combine mfcc with output of CNN
        x = torch.unsqueeze(x,0)

        # Pass through RNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.attention(x, torch.IntTensor([1]).to(device))
        x = self.layer3(x)
        x = torch.unsqueeze(x, 0)
        return x


# Initialize the CRNN, loss, and optimizer
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
        inputs = numpy_array
        labels = torch.from_numpy(np.array([i])) # label the training data with index as the label
        labels.to(device)
        train.append((inputs,labels))

print("Shuffling training data")
random.shuffle(train) # shuffling is important for training
print("training data length: ", len(train))
print()




# Get test data
print("Getting testing data")
root_path = '../../features/mfcc_pitch/'
langs = ['EN', 'ES', 'FR', 'KR', 'MD', 'RS', 'TR', 'BN', 'JV'] # All features for all 9 languages (7 in-set and 2 out-of-set)
file_name = 'raw_mfcc_pitch_test.1.ark'

test = []
for i,lang in enumerate(langs,0): # for each language
    for key, numpy_array in kaldiio.load_ark(root_path + lang + '/' + file_name): # read the ark file
        inputs = numpy_array
        labels = torch.from_numpy(np.array([i])).to(device)
        if lang == 'BN' or lang == 'JV': # Out-of-set languages will have label -1
            labels = torch.from_numpy(np.array([-1])).to(device)
        test.append((inputs,labels))
print("test data length: ", len(test))
print()




# Method for running the test data through model for various thresholds
def test_model(net, test):
    print("Testing...")
    for threshold in [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]: # Testing different thresholds
        correct_per_set = {'in': 0, 'out': 0} # counts correct for in-set and out-of-set
        total_per_set = {'in': 0, 'out': 0} # counts the total test data points
        correct = 0 # correct (used for overall accuracy)
        total = 0 # total (used for overall accuracy)
        with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for i, data in enumerate(test,0):
                inputs, labels = data[0], data[1].to(device)
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
                if labels.to(device) == torch.from_numpy(np.array([-1])).to(device):
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




# Train the CRNN network
do_training = True # Whether to train or load a saved model
SAVE_PATH = '../../saved_models/crnna'
LOAD_PATH = '../../saved_models/crnna__.pth' # pick an epoch to load the model from

if do_training:
    print('Started Training')

    for epoch in range(24):  # number of epochs
        random.shuffle(train) # shuffle data every epoch
        print('\tepoch: ', str(epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            inputs, labels = data[0], data[1].to(device)
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
    
        torch.save(net.state_dict(), SAVE_PATH + str(epoch) + '.pth') # Save the model
        if (epoch+1) % 4 == 0: # Run test every 4 epochs
            test_model(net, test)

    print('Finished Training')
else:
    # Skip training and load a saved model
    print('Skip training')
    print('Loading the model')
    net = Net()
    net.to(device)
    net.load_state_dict(torch.load(LOAD_PATH, map_location=device))
    print('Finished loading')
    print()
    test_model(net, test)
