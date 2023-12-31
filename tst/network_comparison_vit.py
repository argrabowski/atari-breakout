import dqn
from atari_wrappers import make_atari,wrap_deepmind
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformer_package.models.transformer import ViT
from tqdm import tqdm

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


def create_168x168_images(batch_frames):
    """
    Create a batch of 168x168 images by arranging sets of four 84x84 frames in a square configuration for each sample in the batch.

    Args:
    batch_frames (Tensor): A tensor containing a batch of sets of four 84x84 frames.
                           Expected shape: [batch_size, 4, 84, 84]

    Returns:
    Tensor: A batch of single 168x168 images, each combining four frames.
    """
    batch_size = batch_frames.size(0)
    combined_frames = []

    for i in range(batch_size):
        frames = batch_frames[i]
        top_row = torch.cat((frames[0], frames[1]), dim=1)  # Concatenate F1 and F2 horizontally
        bottom_row = torch.cat((frames[2], frames[3]), dim=1)  # Concatenate F3 and F4 horizontally
        combined_frame = torch.cat((top_row, bottom_row), dim=0)  # Concatenate top and bottom vertically
        combined_frames.append(combined_frame.unsqueeze(0))  # Add batch dimension

    return torch.cat(combined_frames, dim=0)  # Combine all samples in the batch



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
target_network = ViT(image_size=168, channel_size=1, patch_size=7, embed_size=512, num_heads=8, classes=4,
                     num_layers=3, hidden_size=256, dropout=0.2).to(device)
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
    num_batches = len(memory) // batch_size

    # Shuffle the indices of the memory
    indices = np.arange(len(memory))
    np.random.shuffle(indices)

    # Initialize tqdm progress bar
    with tqdm(total=num_batches, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
        for batch_start in range(0, len(memory), batch_size):
            # Extract a batch of states
            state_batch = torch.stack(
                [memory.states[i][:4] for i in range(batch_start, min(batch_start + batch_size, len(memory)))])
            state_batch = state_batch.to(device)

            # Process state batch for policy_network
            policy_state_batch = state_batch  # Assuming policy_network uses standard DQN input format

            # Turn off gradients for policy_network
            with torch.no_grad():
                policy_output = policy_network(policy_state_batch)

            # Reset gradients in the optimizer
            optimizer.zero_grad()

            # Compute the output of the target_network (ViT)
            # Process state batch for target_network (ViT)
            vit_state_batch = create_168x168_images(state_batch)
            vit_state_batch = vit_state_batch.to(device)

            # Ensure the input tensor has 4 dimensions: [batch_size, channels, height, width]
            # Add a channel dimension if necessary
            if vit_state_batch.dim() == 3:
                vit_state_batch = vit_state_batch.unsqueeze(1)  # Add channel dimension

            vit_state_batch = vit_state_batch.float()  # Convert to float

            # Now pass the reshaped tensor to the target_network
            target_output = target_network(vit_state_batch)

            # Compute loss
            loss = criterion(target_output, policy_output)

            # Backpropagation
            loss.backward()

            # Update target_network parameters
            optimizer.step()

            # Update total loss and progress bar
            total_loss += loss.item()
            avg_loss = total_loss / ((batch_start // batch_size) + 1)
            pbar.set_postfix(avg_loss=f'{avg_loss:.4f}')
            pbar.update(1)

    # Print average loss per epoch
    avg_loss = total_loss / num_batches
    print(f"Average Loss: {avg_loss:.4f}")
    # Save the state of the target_network
    model_save_path = os.path.join("../trained_policies/previously_trained_policies", f"vit.pt")
    torch.save(target_network.state_dict(), model_save_path)
    print(f"Saved target_network state at: {model_save_path}")

