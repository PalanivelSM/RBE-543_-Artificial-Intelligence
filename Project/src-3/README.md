# Deep Reinforcement Learning for Lunar Landing

This project trains an agent to land a spacecraft on the moon using Deep Q-Learning, Monte Carlo Tree Search (MCTS), and Model Predictive Control (MPC). The agents are evaluated in the Lunar Lander environment from OpenAI Gym.

## Installation

Follow the steps below to set up the environment and install the required libraries:

Install the required libraries:
   ```bash
   pip install gymnasium
   pip install "gymnasium[atari, accept-rom-license]"
   apt-get install -y swig  # For Linux users
   pip install "gymnasium[box2d]"
   pip install torch numpy
   ```

## Running the Project

To run the project, execute the following command:

```bash
python src/main.py
```

You will be prompted to select an agent type (`DQN`, `MCTS`, or `MPC`). The program will train the selected agent and display the results.

## Visualizing Results

After training, the program will visualize the agent's performance in the Lunar Lander environment. For MCTS agents, it will replay the best action sequence found during training. This can be viewed in the video.mp4 file.

## Notes

- Ensure you have Python 3.7 or higher installed.
- If you encounter any issues with dependencies, refer to the official documentation of the respective libraries.
- 