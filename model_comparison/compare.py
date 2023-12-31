import torch
import dqn as dqn
from atari_wrappers import make_atari,wrap_deepmind
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def render_frames(stacked_frames):
    """
    Renders and displays each frame from a stacked tensor of frames.
    Args:
    stacked_frames (torch.Tensor): A stacked tensor of frames.
    """
    num_frames = stacked_frames.shape[0]
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    for i in range(num_frames):
        frame = stacked_frames[i].cpu().numpy()

        # Reshape the frame to 2D if it's 1D
        if frame.ndim == 1:
            # Assuming the frame is a square image, reshape accordingly
            side_length = int(np.sqrt(len(frame)))
            frame = frame.reshape(side_length, side_length)

        ax = axes[i] if num_frames > 1 else axes
        ax.imshow(frame, cmap='gray')
        ax.set_title(f"Frame {i + 1}")
        ax.axis('off')
    plt.show()

def load_pretrained_model(model, device, policy_path=""):
    # Check if a pre-trained model exists at the specified path
    if os.path.isfile(policy_path):
        # If a pre-trained model exists, load its state dictionary into the input model
        # and ensure it's loaded onto the correct device
        print("Loading pre-trained model from", policy_path)
        model.load_state_dict(torch.load(policy_path, map_location=device))
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

#########################################################################################
#########################################################################################
#########################################################################################


policy_network = dqn.DQN(possible_actions, device).to(device) # Create instance of DQN

# Load the 420 points model
policy_network = load_pretrained_model(policy_network,
                                       device,
                                       policy_path="../trained_policies/previously_trained_policies/420_points_policy.pt")


#########################################################################################
#########################################################################################
#########################################################################################


target_network = dqn.DQN(possible_actions, device).to(device) # Create instance of DQN
# Load the 20 points model
target_network = load_pretrained_model(target_network,
                                       device,
                                       policy_path="../trained_policies/previously_trained_policies/20_points_policy.pt")


#########################################################################################
#########################################################################################
#########################################################################################

# Replace these values with the actual values from your environment or saved state
memory_size = 500000  # Example capacity
frame_stack_size = 5  # Number of frames in the state
height, width = 84, 84  # Dimensions of each frame
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory = dqn.ReplayMemory(memory_size, [frame_stack_size, height, width], device)

memory.load_replay_memory(load_path="../replay_buffer/replay_memory_5.pt")

# Assuming the first state in the replay memory is stored as a sequence of frames
# Extract the first state (5 frames)
first_state = memory.states[0]
# Prepare the first state for input into the network, using only the first 4 frames
first_state_input = first_state[:4].unsqueeze(0)  # Use only the first 4 frames and add a batch dimension

print("-----------------------------")
print("Testing phase")
print("-----------------------------")

# Loop over all states in the replay memory
for i in range(len(memory)):
    # Extract each state (using only the first 4 frames)
    state = memory.states[i][:4].unsqueeze(0)  # Add a batch dimension

    # Feed the state into the DQN
    with torch.no_grad():  # Turn off gradients for inference
        output = policy_network(state.to(device))
        target_output = target_network(state.to(device))

    # Print only the numerical output array of the DQN for this state
    print(f"Output of the DQN for state {i}:", output.cpu().numpy())
    print(f"Output of the Transformer DQN for state {i}:", target_output.cpu().numpy())
    print("-----------------------------")