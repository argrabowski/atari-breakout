#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
        


# In[42]:


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

encoder = Encoder()
decoder = Decoder()

# Example forward pass
x = torch.randn(1, 1, 84, 84, 4)  # Example input with a depth dimension
skip1, skip2, bottleneck = encoder(x)



output = decoder( skip1, skip2, bottleneck)


#print(output)


# In[43]:


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
train(encoder, decoder, batch_size=20, num_epochs=10, learning_rate=0.001)


# In[ ]:




