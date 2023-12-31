"""
Script Name: s2_mlp.py

Neural network architecture for the Spatial-Shift MLP (S2-MLP) algorithm, used for training an agent to play a game from pixel input.
The agent learns to play the game using reinforcement learning by interacting with the environment and updating the weights of the
neural network accordingly. The script also defines an epsilon-greedy action selection strategy, a replay memory buffer,
and a function for converting input image frames to PyTorch tensors. The replay memory buffer stores experience tuples
that the agent uses to update its Q-values, and the epsilon-greedy strategy balances exploration and exploitation during training.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from timm.models.layers import PatchEmbed
from einops.layers.torch import Reduce

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels):
        super(LayerNorm2d, self).__init__(num_channels)

    def forward(self, x):
        # Permute dimensions for LayerNorm computation
        x = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute back to original order
        return x.permute(0, 3, 1, 2)

class SpatialShift(nn.Module):
    def __init__(self):
        super(SpatialShift, self).__init__()

    def forward(self, x):
        # Shift spatial dimensions of input tensor
        b, w, h, c = x.size()
        x[:, 1:, :, :c//4] = x[:, :w-1, :, :c//4]
        x[:, :w-1, :, c//4:c//2] = x[:, 1:, :, c//4:c//2]
        x[:, :, 1:, c//2:c*3//4] = x[:, :, :h-1, c//2:c*3//4]
        x[:, :, :h-1, 3*c//4:] = x[:, :, 1:, 3*c//4:]
        return x

class S2Block(nn.Module):
    def __init__(self, dim, expand_ratio=1, mlp_bias=True):
        super(S2Block, self).__init__()
        # Two sequential MLPs for channel-wise processing
        self.channel_mlp1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=mlp_bias),
            nn.GELU(),
            SpatialShift(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=mlp_bias),
            LayerNorm2d(dim))
        self.channel_mlp2 = nn.Sequential(
            nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, stride=1, bias=mlp_bias),
            nn.GELU(),
            nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, stride=1, bias=mlp_bias),
            LayerNorm2d(dim))

    def forward(self, x):
        # Residual connection with output of two MLPs
        x = self.channel_mlp1(x) + x
        x = self.channel_mlp2(x) + x
        return x

class S2MLP(nn.Module):
    def __init__(self, device, num_actions, img_size=84, patch_size=6, in_chans=4, embed_dim=384, depth=36, expand_ratio=4, mlp_bias=True):
        super(S2MLP, self).__init__()
        # Patch embedding layer
        self.patch_emb = PatchEmbed(img_size, patch_size, in_chans, embed_dim, flatten=False)
        # Stacked S2Blocks as main processing stages
        self.stages = nn.Sequential(
            *[S2Block(embed_dim, expand_ratio, mlp_bias) for _ in range(depth)])
        # MLP head for final classification
        self.mlp_head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(embed_dim, num_actions))
        self.device = device

    def forward(self, x):
        # Normalize input and pass through S2MLP model
        x = x.to(self.device).float() / 255.
        x = self.patch_emb(x)
        x = self.stages(x)
        out = self.mlp_head(x)
        return out.to(self.device)

    def init_weights(self, m):
        # Initialize weights using Kaiming normal initialization
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Define the epsilon-greedy action selection strategy
class ActionSelector(object):
    def __init__(self,
                 initial_eps=None,
                 final_eps=None,
                 eps_decay=None,
                 random_exp=None,
                 policy_net=None,
                 n_actions=None,
                 dev=None,
                 for_evaluation=False):

        # If for_evaluation flag is True, then initialize for evaluation.
        if for_evaluation:
            self.eps = initial_eps
            self.policy_network = policy_net
            self.possible_actions = n_actions
            self.device = dev
            return

        # Otherwise, initialize normally.
        self.eps = initial_eps
        self.final_epsilon = final_eps
        self.initial_epsilon = initial_eps
        self.random_exploration_interval = random_exp
        self.policy_network = policy_net
        self.epsilon_decay = eps_decay
        self.possible_actions = n_actions
        self.device = dev

    def select_action(self, state, current_episode=1, training=False):
        sample = random.random()
        if training:
            # Update the value of epsilon during training based on the current episode
            self.eps = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * \
                       math.exp(-1. * (current_episode-self.random_exploration_interval) / self.epsilon_decay)
            self.eps = max(self.eps, self.final_epsilon)
        if sample > self.eps:
            with torch.no_grad():
                a = self.policy_network(state).max(1)[1].cpu().view(1, 1)
        else:
            a = torch.tensor([[random.randrange(self.possible_actions)]], device=self.device, dtype=torch.long)
        return a.cpu().numpy()[0, 0].item(), self.eps

# Define the replay memory data structure for storing experience tuples
class ReplayMemory(object):
    def __init__(self, capacity, state_shape, device):
        replay_buffer_capacity,height,width = state_shape
        # Store the capacity of the memory buffer, the device, and initialize the memory buffer
        self.capacity = capacity
        self.device = device
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.states = torch.zeros((capacity, replay_buffer_capacity, height, width), dtype=torch.uint8)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool)
        # Initialize the position and size of the memory buffer
        self.position = 0
        self.size = 0

    # Add a new experience tuple to the memory buffer
    def push(self, state, action, reward, done):
        self.states[self.position] = state
        self.actions[self.position,0] = action
        self.rewards[self.position,0] = reward
        self.dones[self.position,0] = done
        # Update the position and size of the memory buffer
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    # Sample a batch of experiences from the memory buffer
    def sample(self, batch_state):
        i = torch.randint(0, high=self.size, size=(batch_state,))
        # Get the current state and next state
        batch_next_state = self.states[i, 1:]
        batch_state = self.states[i, :4]
        # Get the action, reward, and done values for each experience tuple in the batch
        batch_reward = self.rewards[i].to(self.device).float()
        batch_done = self.dones[i].to(self.device).float()
        batch_actions = self.actions[i].to(self.device)
        # Return the batch of experiences
        return batch_state, batch_actions, batch_reward, batch_next_state, batch_done

    # Return the current size of the memory buffer
    def __len__(self):
        return self.size

    def shuffle(self):
        # Get the current size of the memory
        memory_size = len(self)

        # Create a random permutation of indices
        permutation = torch.randperm(memory_size)

        # Shuffle all elements of the memory
        self.states = self.states[permutation]
        self.actions = self.actions[permutation]
        self.rewards = self.rewards[permutation]
        self.dones = self.dones[permutation]

    def save_replay_memory(self, save_path='replay_memory.pt'):
        # Create a dictionary to hold the replay memory contents
        memory_dict = {
            'actions': self.actions[:self.size],
            'states': self.states[:self.size],
            'rewards': self.rewards[:self.size],
            'dones': self.dones[:self.size]
        }

        # Save the dictionary using PyTorch's save function
        torch.save(memory_dict, save_path)

    def load_replay_memory(self, load_path='replay_memory.pt'):
        if os.path.isfile(load_path):
            # Load the saved replay memory
            memory_dict = torch.load(load_path)

            # Load each component of the replay memory
            self.actions = memory_dict['actions']
            self.states = memory_dict['states']
            self.rewards = memory_dict['rewards']
            self.dones = memory_dict['dones']

            # Update the position and size of the memory buffer
            self.size = len(self.actions)
            self.position = self.size % self.capacity
        else:
            print(f"Replay memory file {load_path} not found. Starting with an empty replay memory.")

def get_frame_tensor(pixels):
    pixels = torch.from_numpy(pixels)
    height = pixels.shape[-2]
    return pixels.view(1, height, height)
