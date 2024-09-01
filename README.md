# Reinforcement Learning for Atari Breakout

This project implements Deep Q-Network (DQN) and other advanced architectures using PyTorch for training an Artificial Intelligence agent to play the classic Atari game Breakout. The agent uses Reinforcement Learning to learn and improve its performance in the game environment.

https://github.com/user-attachments/assets/c9e576d5-ad01-420b-8bfa-28956a9e7b1e

## Components

### 1. Architecture

- Definition of the Deep Q-Network (DQN), Spatial-Shift MLP (S2-MLP), and Convolutional U-Net using PyTorch.
- Takes the state as input and outputs Q-values for each possible action.
- Training using the Bellman equation and the Huber loss.

### 2. Replay Memory

- Definition of a ReplayMemory class for storing experience tuples (state, action, reward, done, next_state).
- Helps break the sequential correlation between experiences and provides a diverse set for training.

### 3. Atari Wrappers

- Set of wrappers for modifying the behavior of Atari environments in OpenAI Gym.
- Wrappers include preprocessing frames, handling episodic life, frame skipping, reward clipping, frame stacking, etc.

### 4. Agent Trainer

- Main training loop for the RL agent.
- Hyperparameters control the training process and can be adjusted for performance.
- Periodic evaluation of the agent's performance and model saving.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/argrabowski/atari-breakout.git
   ```

2. Run the renderer (420 points model):

   ```bash
   cd 420_points_model
   python renderer.py
   ```

## Dependencies

- Python 3.10
- PyTorch
- OpenAI Gym
- OpenCV
