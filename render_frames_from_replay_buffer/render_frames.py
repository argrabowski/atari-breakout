import numpy as np
import matplotlib.pyplot as plt
import torch

# Function to render and display each frame from a stacked tensor of frames
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

# Load the replay memory from a file
replay_memory_path = '../replay_buffer/replay_memory_5.pt'  # Replace with your file path
replay_memory = torch.load(replay_memory_path)

# Extract the first state from the replay memory
first_state = replay_memory['states'][0]

# Render and display the frames of the first state
render_frames(first_state)
