#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import torch

#with h5py.File('C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\state_data.h5', 'w') as 
#state_hdf, h5py.File('C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\action_data.h5', 'w') as action_hdf:

"""
# Read states from the HDF5 file
with h5py.File('C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\state_data.h5', 'r') as state_hdf:
    # Iterate over each dataset in the file
    for key in state_hdf.keys():
        # Load the dataset as a numpy array
        state_array = state_hdf[key][...]
        # Convert the numpy array to a PyTorch tensor
        state_tensor = torch.tensor(state_array)
        print(f"State {key}: {state_tensor.shape}")
        #print(state_tensor)
        print(state_tensor[0][0][15]
        break

# Read actions from the HDF5 file
with h5py.File('C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\action_data.h5', 'r') as action_hdf:
    # Iterate over each dataset in the file
    for key in action_hdf.keys():
        # Load the dataset as a numpy array
        action_array = action_hdf[key][...]
        # Convert the numpy array to a PyTorch tensor
        action_tensor = torch.tensor(action_array)
        print(f"Action {key}: {action_tensor.shape}")
        print(action_tensor)
        break
        
        
        
"""      
        


# In[2]:


import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)  # Updated in_channels
        self.conv2 = nn.Conv3d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print('x.shape')
        #print(x.shape)
        x1 = self.relu(self.conv1(x))
        #print('x1.shape')
        #print(x1.shape)
        p1 = self.pool(x1)
        #print('p1.shape')
        #print(p1.shape)
        x2 = self.relu(self.conv2(p1))
        #print('x2.shape')
        #print(x2.shape)
        p2 = self.pool(x2)
        #print('p2.shape')
        #print(p2.shape)
        return x1, x2, p2
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv1 = nn.Conv3d(192, 64, 3, padding=1)  # Ensure the channel dimensions match for concatenation
        self.upconv2 = nn.ConvTranspose3d(64, 4, 2, stride=2)
        self.conv2 = nn.Conv3d(68, 1, 3, padding=1)  # Updated output channels to 1 for grayscale
        self.relu = nn.ReLU()

    def forward(self, x1, x2, p2):
        x = self.relu(self.upconv1(p2))
        #print(x.shape)
        x = torch.cat((x2, x), dim=1)
        #print(x.shape)
        x = self.relu(self.conv1(x))
        #print(x.shape)
        x = self.relu(self.upconv2(x))
        #print(x.shape)
        x = torch.cat((x1, x), dim=1)
        #print(x.shape)
        x = self.relu(self.conv2(x))
        #print(x.shape)
        return x

#encoder = Encoder()
#decoder = Decoder()

# Example forward pass


# In[4]:


from torch.utils.data import DataLoader, Dataset
import os
import torch.optim as optim
#from torch.utils.data import DataLoader
import torch.nn as nn

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(file_path, 'r') as file:
            self.keys = list(file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as file:
            data = file[self.keys[index]][...]
        return torch.tensor(data, dtype=torch.float32) / 255.0

def train(encoder, decoder, batch_size, num_epochs, learning_rate):
    encoder_path="C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\W_Encoder"
    decoder_path="C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\W_Decoder"
    
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path))
        print("Loaded encoder weights")

    if os.path.exists(decoder_path):
        decoder.load_state_dict(torch.load(decoder_path))
        print("Loaded decoder weights")
    
    # Load dataset
    dataset = HDF5Dataset('C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\state_data.h5')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Size of the dataset: {len(dataset)}")
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            encoded_states = encoder(batch)
            decoded_states = decoder(*encoded_states)

            # Compute loss
            loss = criterion(decoded_states, batch)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print("Training complete and models saved")

# Example usage
encoder = Encoder()
decoder = Decoder()

if True:
    encoder_path="C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\W_Encoder"
    decoder_path="C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\W_Decoder"
    
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path))
        print("Loaded encoder weights")

    if os.path.exists(decoder_path):
        decoder.load_state_dict(torch.load(decoder_path))
        print("Loaded decoder weights")


if False:
    train(encoder, decoder, batch_size=20, num_epochs=10, learning_rate=0.001)
    
    


# In[5]:


x = torch.randn(1, 1, 84, 84, 4)  # Example input with a depth dimension
skip1, skip2, bottleneck = encoder(x)

#output = decoder( skip1, skip2, bottleneck)
#print(output)


print('skip1.shape')
print(skip1.shape)
print('skip2.shape')
print(skip2.shape)
print('bottleneck.shape')
print(bottleneck.shape)








# In[6]:


flat_skip1 = skip1.view(-1)
flat_skip2 = skip2.view(-1)

# Concatenate the flattened tensors
concatenated = torch.cat([flat_skip1, flat_skip2], dim=0)

print(concatenated.shape)


# In[7]:


import torch.nn as nn
import torch.nn.functional as F



# Define the custom smooth ReLU activation function
def smooth_relu(x):
    return torch.exp(F.relu(x)) + 1

# Define the neural network
class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier, self).__init__()
        self.fc1 = nn.Linear(2257920, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 64)      # Second hidden layer
        self.fc3 = nn.Linear(64, 4)          # Output layer

    def forward(self, x):
        x = smooth_relu(self.fc1(x))
        x = smooth_relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.shape)  # Debugging: Check the shape of x before softmax
        x = F.softmax(x, dim=1)
        return x
    
model = MyClassifier()


# In[8]:


flat_skip1 = skip1.view(-1)
flat_skip2 = skip2.view(-1)

concatenated = torch.cat([flat_skip1, flat_skip2], dim=0).unsqueeze(0)


with torch.no_grad():
    output = model(concatenated)
    
print(output)


# In[9]:


class HDF5Dataset(Dataset):
    def __init__(self, data_file_path, label_file_path):
        super(HDF5Dataset, self).__init__()
        self.data_file_path = data_file_path
        self.label_file_path = label_file_path
        with h5py.File(data_file_path, 'r') as file:
            self.keys = list(file.keys())

    def __len__(self):
        return len(self.keys)

    def get_old(self, index):
        with h5py.File(self.file_path, 'r') as file:
            data = file[self.keys[index]][...]
        return torch.tensor(data, dtype=torch.float32) / 255.0
    
    def __getitem__(self, index):
        with h5py.File(self.data_file_path, 'r') as file:
            data = file[list(file.keys())[index]][...]
        with h5py.File(self.label_file_path, 'r') as file:
            label = file[list(file.keys())[index]][...]
        # Convert label to one-hot vector
        label_one_hot = torch.zeros(4, dtype=torch.float32)
        label_one_hot[int(label)] = 1
        return torch.tensor(data, dtype=torch.float32) / 255.0, label_one_hot


# In[12]:


for param in encoder.parameters():
    param.requires_grad = False


def train_classifier(model, encoder, batch_size, num_epochs, learning_rate):
    # Paths
    model_path = "C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\W_Classifier"

    # Load dataset
    dataset = HDF5Dataset(data_file_path='C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\state_data.h5', label_file_path='C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\action_data.h5')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Size of the dataset: {len(dataset)}")
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total = 0
        correct = 0
        for batch, labels in dataloader:
            # Prepare input for classifier
            skip1, skip2, _ = encoder(batch)
            flat_skip1 = skip1.view(skip1.size(0), -1)
            flat_skip2 = skip2.view(skip2.size(0), -1)
            concatenated = torch.cat([flat_skip1, flat_skip2], dim=1)

            # Forward pass
            outputs = model(concatenated)

            # Compute loss
            loss = criterion(outputs, torch.max(labels, 1)[1])  # Using indices of one-hot vectors
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
            
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy}%')


    torch.save(model.state_dict(), model_path)
    print("Training complete and model saved")
    


# In[13]:


train_classifier(model, encoder, batch_size=20, num_epochs=1, learning_rate=0.001)


# In[13]:


if False:
    data_file_path='C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\state_data.h5'
    label_file_path='C:\\Users\\IIISI\\OneDrive\\Documents\\DLF\\atari_breakout_dqn_pytorch\\dqn\\action_data.h5'

    file_path = data_file_path

    # Open the file
    with h5py.File(file_path, 'r') as file:
        # Iterate through each dataset in the file
        for key in file.keys():
            # Access the dataset
            data = file[key]

            # Print the shape of the dataset
            print(f"Shape of data in '{key}': {data.shape}")
            print(data)


# In[ ]:





# In[33]:


if False:
    with h5py.File(data_file_path, 'r') as file:
        l1=list(file.keys())
        print(list(file.keys()))

    with h5py.File(label_file_path, 'r') as file:
        l2=list(file.keys())
        print(list(file.keys()))
    
    
    

    


# In[ ]:




