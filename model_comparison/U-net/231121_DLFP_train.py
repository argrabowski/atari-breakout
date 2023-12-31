#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=0)
        self.pool = nn.MaxPool3d(1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print('x.shape')
        #print(x.shape)
        x1 = self.relu(self.conv1(x))
        #print('x1.shape')
        #print(x1.shape)
        p1 = self.pool(x1)

        return x1, p1
    
encoder = Encoder()



class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        #self.fc1 = nn.Linear(2257920, 256)  # First hidden layer
        self.fc1 = nn.Linear(53792, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 64)      # Second hidden layer
        self.fc3 = nn.Linear(64, 4)          # Output layer

    def forward(self, x):
        #x = smooth_relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        #x = smooth_relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.shape)  # Debugging: Check the shape of x before softmax
        x = F.softmax(x, dim=-1)
        return x
    
model = MyClassifier()

x = torch.randn(1, 1, 84, 84, 4)  
skip1, bottleneck = encoder(x)

print([skip1.shape, bottleneck.shape])

skip1_flat = skip1.view(skip1.size(0), -1)  # Flattens all dimensions except the batch size
bottleneck_flat = bottleneck.view(bottleneck.size(0), -1)

# Concatenate the flattened tensors
combined = torch.cat((skip1_flat, bottleneck_flat), dim=1)

#print(combined.shape)

output=model(combined)

print(output)


class HDF5Dataset(Dataset):
    def __init__(self, data_file_path, label_file_path):
        super(HDF5Dataset, self).__init__()
        self.data_file_path = data_file_path
        self.label_file_path = label_file_path
        with h5py.File(data_file_path, 'r') as file:
            self.keys = list(file.keys())

    def __len__(self):
        return len(self.keys)


    
    def __getitem__(self, index):
        with h5py.File(self.data_file_path, 'r') as file:
            data = file[list(file.keys())[index]][...]
        with h5py.File(self.label_file_path, 'r') as file:
            label = file[list(file.keys())[index]][...]
        # Convert label to one-hot vector
        label_one_hot = torch.zeros(4, dtype=torch.float32)
        label_one_hot[int(label)] = 1
        return torch.tensor(data, dtype=torch.float32) / 255.0, label_one_hot
    
    
    


# In[22]:

from torch.optim.lr_scheduler import ExponentialLR

data_file_path='C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\state_data.h5'
label_file_path='C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\action_data.h5'

def train_model(model, encoder, train_dataset, epochs, learning_rate):
    model_path = "C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\W_Classifier_2"
    encoder_path = "C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\W_Encoder_2"
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    
    
    
    
    anneal_factor=0.95
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=anneal_factor)

    for epoch in range(epochs):
        model.train()
        encoder.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            skip1, bottleneck = encoder(inputs)
            skip1_flat = skip1.view(skip1.size(0), -1)
            bottleneck_flat = bottleneck.view(bottleneck.size(0), -1)
            combined = torch.cat((skip1_flat, bottleneck_flat), dim=1)

            outputs = model(combined)
            loss = criterion(outputs, labels)
            
            #print([outputs.shape, labels.shape])
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            
            #print(predicted)
            #print(torch.argmax(labels,dim=1))
            
            total_predictions += labels.size(0)
            correct_predictions += (predicted == torch.argmax(labels,dim=1)).sum().item()
            


        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        scheduler.step()
    torch.save(model.state_dict(), model_path)
    torch.save(encoder.state_dict(), encoder_path)
    print('Training complete')

    
train_dataset = HDF5Dataset(data_file_path,label_file_path)
print(["train_dataset",len(train_dataset)])
train_model(model, encoder, train_dataset, 40, 0.002)







