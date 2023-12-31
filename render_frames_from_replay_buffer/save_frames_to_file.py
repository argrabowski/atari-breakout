import numpy as np
import matplotlib.pyplot as plt
import torch

# Function to save each frame from a stacked tensor of frames as separate image files
def save_frames(stacked_frames, base_filename="frame"):
    """
    Saves each frame from a stacked tensor of frames as separate image files.
    Args:
    stacked_frames (torch.Tensor): A stacked tensor of frames.
    base_filename (str): Base filename for saving each frame.
    """
    num_frames = stacked_frames.shape[0]
    for i in range(num_frames):
        frame = stacked_frames[i].cpu().numpy()

        # Reshape the frame to 2D if it's 1D
        if frame.ndim == 1:
            # Assuming the frame is a square image, reshape accordingly
            side_length = int(np.sqrt(len(frame)))
            frame = frame.reshape(side_length, side_length)

        # Save the frame as an image file
        plt.imshow(frame, cmap='gray')
        plt.title(f"Frame {i + 1}")
        plt.axis('off')
        plt.savefig(f"{base_filename}_{i+1}.png")
        plt.close()

# Load the replay memory from a file
replay_memory_path = '../replay_buffer/replay_memory_5.pt'  # Replace with your file path
replay_memory = torch.load(replay_memory_path)

# Extract the first state from the replay memory
first_state = replay_memory['states'][0]

# Save the frames of the first state as separate files
save_frames(first_state, "first_state_frame")
