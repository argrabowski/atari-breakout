import dqn
from atari_wrappers import make_atari,wrap_deepmind
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ... [Previous Code for Setup and Loading Models] ...
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
pretrained_policy_path = "../trained_policies/previously_trained_policies/420_points_policy.pt"

def load_pretrained_model(model, device):
    # Check if a pre-trained model exists at the specified path
    if os.path.isfile(pretrained_policy_path):
        # If a pre-trained model exists, load its state dictionary into the input model
        # and ensure it's loaded onto the correct device
        print("Loading pre-trained model from", pretrained_policy_path)
        model.load_state_dict(torch.load(pretrained_policy_path, map_location=device))
    else:
        # If a pre-trained model does not exist, initialize the input model's weights from scratch
        print("Pre-trained model not found. Training from scratch.")
        model.apply(model.init_weights)
    # Return the input model (either loaded with a pre-trained model or initialized from scratch)
    return model


max_episode_steps = 500000

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPU if available
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Set the device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the name of the game environment
env_name = 'Breakout'
# Create a raw Atari environment without frame skipping
env_raw = make_atari('{}NoFrameskip-v4'.format(env_name), max_episode_steps, False)

# Wrap the Atari environment in a DeepMind wrapper
env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

# Get the replay buffer capacity, height, and width from the first frame
replay_buffer_capacity, height, width = dqn.get_frame_tensor(env.reset()).shape

# Get the number of possible actions in the game
possible_actions = env.action_space.n

policy_network = dqn.DQN(possible_actions, device).to(device)
policy_network = load_pretrained_model(policy_network, device)
target_network = dqn.DQN(possible_actions, device).to(device)
#pretrained_policy_path = "prueba.pt"
#target_network = load_pretrained_model(target_network, device)

# Replace these values with the actual values from your environment or saved state
memory_size = 500000  # Example capacity
frame_stack_size = 5  # Number of frames in the state
height, width = 84, 84  # Dimensions of each frame
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory = dqn.ReplayMemory(memory_size, [frame_stack_size, height, width], device)

memory.load_replay_memory(load_path="../replay_buffer/replay_memory_500k.pt")

# Criterion (Loss Function)
criterion = nn.MSELoss()

# Optimizer for the target_network
optimizer = optim.Adam(target_network.parameters(), lr=0.001)

# Batch size
batch_size = 32

# Number of training epochs
num_epochs = 10  # Adjust this number based on your needs

# Training Loop
for epoch in range(num_epochs):
    total_loss = 0

    # Shuffle the indices of the memory
    indices = np.arange(len(memory))
    np.random.shuffle(indices)

    for batch_start in range(0, len(memory), batch_size):
        # Extract a batch of states
        state_batch = torch.stack([memory.states[i][:4] for i in range(batch_start, min(batch_start + batch_size, len(memory)))])
        state_batch = state_batch.to(device)

        # Turn off gradients for policy_network
        with torch.no_grad():
            policy_output = policy_network(state_batch)

        # Reset gradients in the optimizer
        optimizer.zero_grad()

        # Compute the output of the target_network
        target_output = target_network(state_batch)

        # Compute loss
        loss = criterion(target_output, policy_output)

        # Backpropagation
        loss.backward()

        # Update target_network parameters
        optimizer.step()

        total_loss += loss.item()

    # Print average loss per epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / (len(memory) // batch_size)}")
