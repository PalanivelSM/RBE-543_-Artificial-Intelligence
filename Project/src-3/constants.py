## Deep Learning Hyperparameters
LEARNING_RATE = 5e-4
MINIBATCH_SIZE = 100
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = int(1e5)
INTERPOLATION_PARAMETER = 1e-3

## Model Predictive Control Hyperparameters
MIN_PLAN_HORIZON = 1  # Number of steps to simulate in the future from the current state (minimum)
MAX_PLAN_HORIZON = 10  # Number of steps to simulate in the future from the current state (maximum)
HORIZON_INCREMENT = 1  # How much to increase the horizon gradually
HORIZON_UPDATE_INTERVAL = 100000  # How often to update the horizon (in steps)
HORIZON_SAMPLES = 10  # How many horizons to try for each current state

## General Training Hyperparameters
NUMBER_EPISODES = 2000
MAXIMUM_NUMBER_TIMESTEPS_PER_EPISODE = 1000
EPSILON_STARTING_VALUE  = 1.0
EPSILON_ENDING_VALUE  = 0.01
EPSILON_DECAY_VALUE  = 0.995
EPSILON = EPSILON_STARTING_VALUE
