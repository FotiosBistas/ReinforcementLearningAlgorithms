import argparse
import pytest
import logging
import numpy as np

from typing import Tuple, List, Callable
from tqdm import trange

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("INFO")

actions = np.array([
    np.array([-1, 0]),  # Decrease vertical speed (UP)
    np.array([1, 0]),   # Increase vertical speed (DOWN)
    np.array([0, -1]),  # Decrease horizontal speed (LEFT)
    np.array([0, 1]),   # Increase horizontal speed (RIGHT)
    np.array([-1, -1]), # Decrease both (UP LEFT)
    np.array([-1, 1]),  # Decrease vertical, increase horizontal (UP RIGHT)
    np.array([1, -1]),  # Increase vertical, decrease horizontal (DOWN LEFT)
    np.array([1, 1]),   # Increase both (DOWN RIGHT)
    np.array([0, 0]),   # No acceleration
])

# Define track size
rows, cols = 32, 17 
MAX_SPEED = 4

# Initialize track grid with zeros (open track)
track = np.zeros((rows, cols), dtype=np.int8)

# Define boundaries (walls) with 1s
track[31, 0:3] = 1 
track[30, 0:2] = 1
track[29, 0:2] = 1
track[28, 0] = 1
track[0:18, 0] = 1
track[0:10, 1] = 1
track[0:3, 2] = 1
track[0:26, 9:] = 1
track[25, 9] = 0  


track = np.flipud(track)

# Define start and finish line positions
start_cols = list(range(3, 9))  # Columns 4 to 9
start_cells = [
    np.array([track.shape[0] - 1, i]) for i in start_cols
]
fin_cells = [
    np.array([i, cols-1]) for i in range(0, 6)
]  # Finish line cells

_logger.info(f"Start cells: {start_cells}")
_logger.info(f"Fin cells: {fin_cells}")

# Convert finish line positions to track representation
for r, c in fin_cells:
    track[r, c] = 2  # Mark finish cells with 2

# Print the track for debugging
_logger.info(f"Track:\n{track}")

def filter_states(
    current_velocity: np.array,
    all_actions: List[np.array],
) -> List[np.array]:
    """
    Filters actions to ensure the resulting velocity:
    1. Has no negative components.
    2. Stays within the valid bounds (0 to MAX_SPEED).
    3. Does not result in a velocity of [0, 0].

    Parameters:
    - current_velocity: np.array -> The current velocity of the agent.
    - all_actions: List[np.array] -> A list of possible actions.

    Returns:
    - List[np.array] -> A filtered list of valid actions.
    """
    filtered_actions = [
        action for action in all_actions 
        if np.all((current_velocity + action) >= 0)           # No negative velocity components
        and np.all((current_velocity + action) <= MAX_SPEED)  # Velocity within MAX_SPEED
        and not np.array_equal(current_velocity + action, [0, 0])  # Resulting velocity != [0, 0]
    ]

    return filtered_actions

def create_episode(
    epsilon: float,
    Q: np.array,
    track: np.array, 
    actions: List[np.array],
    start_cells: List[np.array],
    behavior_policy: Callable,
):
    episode_states = []
    episode_actions = []
    episode_action_probability = []
    episode_rewards = []

    episode_velocity = np.array([0, 0])

    # Initial states are in the start of the Track array
    initial_state = start_cells[np.random.choice(len(start_cells))]
    _logger.debug(f"Starting from state: {initial_state}")

    valid_actions = filter_states(
        current_velocity=episode_velocity, 
        all_actions=actions,    
    )

    initial_action, action_prob = behavior_policy(
        epsilon=epsilon, 
        q_values=Q[initial_state[0], initial_state[1], :], 
        actions=valid_actions,
    )

    episode_states.append(initial_state)
    episode_actions.append(initial_action)
    episode_action_probability.append(action_prob)

    state, episode_velocity, reward = step(
        state=initial_state,
        velocity=episode_velocity, 
        action=initial_action,
        track=track,
        start_cells=start_cells,
    )

    while track[state[0], state[1]] != 2: 
        episode_rewards.append(reward)
        episode_states.append(state)

        valid_actions = filter_states(
            current_velocity=episode_velocity, 
            all_actions=actions,    
        )

        action, action_prob = behavior_policy(
            epsilon=epsilon,
            q_values=Q[state[0], state[1], :],
            actions=valid_actions,
        )

        episode_actions.append(action)
        episode_action_probability.append(action_prob)

        state, episode_velocity, reward = step(
            state=state,
            velocity=episode_velocity, 
            action=action,
            track=track,
            start_cells=start_cells,
        )

    # Append the last reward of the Terminal State
    episode_rewards.append(reward)

    return episode_states, episode_actions, episode_rewards, episode_action_probability

def run(args): 
    global actions
    global track
    global start_cells
    global fin_cells

    max_episodes = args.max_episodes
    discount_factor = args.discount_factor
    epsilon = args.epsilon

    Q = np.ones((rows, cols, len(actions))) * -200
    Counts = np.zeros((rows, cols, len(actions)))
    p_s = greedy
    b = e_soft 
    #b = uniform

    timesteps_per_episode = []  # Track timesteps per episode
    division_epsilon = 1e-8  # Small value to prevent division by zero

    for episode in trange(max_episodes):
        episode_states, episode_actions, episode_rewards, episode_action_probability = create_episode(
            epsilon=epsilon, 
            Q=Q, 
            actions=actions,
            track=track,
            start_cells=start_cells,
            behavior_policy=b,
        )

        timesteps_per_episode.append(len(episode_states))  # Record timesteps

        _logger.info(f"Created episode successfully")

        G = 0 
        W = 1

        for i in reversed(range(len(episode_states))):
            G = discount_factor * G + episode_rewards[i]
            action_index = get_action_index(episode_actions[i])
            Counts[episode_states[i][0], episode_states[i][1], action_index] += W 
            Q[episode_states[i][0], episode_states[i][1], action_index] += (
                W / (Counts[episode_states[i][0], episode_states[i][1], action_index] + division_epsilon)
            ) * (G - Q[episode_states[i][0], episode_states[i][1], action_index])

            best_action = p_s(Q[episode_states[i][0], episode_states[i][1], :], actions=actions)
            _logger.debug(f"Best action using greedy policy: {best_action}")
            _logger.debug(f"Current action: {episode_actions[i]}")
            if np.array_equal(episode_actions[i], best_action): 
                _logger.info("Breaking from inner loop. Current action is the best action.")
                break

            W *= episode_action_probability[i]

        # Plot trajectory for the current episode
        if episode == max_episodes - 1:  # Plot for the last episode
            plot_trajectory(Q=Q, actions=actions, track=track, start_cells=start_cells, file_name=f"trajectory_refined.png")

    # Plot cumulative timesteps vs episodes after training
    plot_cumulative_timesteps(timesteps_per_episode, file_name="cumulative_timesteps.png")


def step(
    state: Tuple[int, int], 
    velocity: Tuple[int, int], 
    action: Tuple[int,int], 
    track: np.array,
    start_cells: List[np.array],
) -> Tuple[Tuple[int, int], Tuple[int, int], int]:

    # Convert inputs to NumPy arrays for easy operations
    state = np.array(state)
    velocity = np.array(velocity)
    action = np.array(action)

    # Update velocity with acceleration
    new_velocity = velocity + action
    new_velocity = np.clip(new_velocity, 0, MAX_SPEED)

    _logger.debug(f"Current state: {state}")
    _logger.debug(f"Updated velocity from: {velocity} to {new_velocity}")

    # Update position with new velocity
    new_state = np.array([state[0] - new_velocity[0], state[1] + new_velocity[1]])

    # **Check if the car has reached the finish line or has surpassed it**
    if track[new_state[0], min(new_state[1], track.shape[1] - 1)] == 2:
        _logger.debug("FINISH LINE REACHED!")
        new_state = np.array([np.clip(new_state[0], 0, track.shape[0] - 1),np.clip(new_state[1], 0, track.shape[1] - 1)])
        return new_state, new_velocity, 0  # Reached finish line

    # **Return immediately if out of bounds**
    if not (0 <= new_state[0] < track.shape[0]) or not (0 <= new_state[1] < track.shape[1]):
        new_start = start_cells[np.random.choice(len(start_cells))]
        new_velocity[:] = 0
        _logger.debug(f"OUT OF BOUNDS DETECTED: {new_state}. Resetting to start position {new_start} with new velocity {new_velocity}.")
        return new_start, new_velocity, -100

    # **Check for collision with a wall**
    if track[new_state[0], new_state[1]] == 1:
        new_start = start_cells[np.random.choice(len(start_cells))]
        new_velocity[:] = 0 
        _logger.debug(f"COLLISION detected at {new_state}. Resetting to start position {new_start} with new velocity {new_velocity}.")
        return new_start, new_velocity, -100  # Large negative reward

    # **Normal step with small penalty for each move**
    _logger.debug(f"VALID MOVE: Moving to {new_state} from {state}. Continuing the race.")
    return new_state, new_velocity, -1

def e_soft(
    epsilon: float, 
    q_values: np.array,
    actions: List[np.array],
):
    """
    ε‑soft policy using filtered Q‑values.
    """
    # Map each filtered action to its index in the full action set
    valid_indices = [get_action_index(a) for a in actions]
    
    # Extract the corresponding Q‑values for these valid actions
    valid_q_values = q_values[valid_indices]
    
    # Create probability distribution for the filtered actions only
    num_valid = len(actions)
    probabilities = np.ones(num_valid) * (epsilon / num_valid)
    
    # Find the index (within valid_q_values) of the best action
    best_local_index = np.argmax(valid_q_values)
    probabilities[best_local_index] += 1 - epsilon
    
    # Choose an action index from the filtered set
    chosen_local_index = np.random.choice(num_valid, p=probabilities)
    chosen_action = actions[chosen_local_index]
    
    return chosen_action, probabilities[chosen_local_index]

def greedy(
    q_values: np.array,
    actions: List[np.array],
) -> Tuple[np.array, float]:
    """
    Greedy policy: Returns the action with the highest Q-value (from the filtered list)
    and its probability (which is 1 since it is a greedy choice).
    """
    # Map each filtered action to its index in the full action set
    valid_indices = [get_action_index(a) for a in actions]
    # Extract the corresponding Q-values for these valid actions
    valid_q_values = q_values[valid_indices]
    # Find the index (within valid_q_values) of the best action
    best_local_index = np.argmax(valid_q_values)
    chosen_action = actions[best_local_index]
    return chosen_action, 1.0

def uniform(
    actions: List[np.array],
) -> Tuple[np.array, float]:
    """
    Uniform policy: Returns an action uniformly at random from the filtered list,
    along with the probability of choosing that action.
    """
    num_valid = len(actions)
    # Create a uniform probability distribution over the filtered actions
    probabilities = np.ones(num_valid) * (1 / num_valid)
    # Sample an index from the filtered list
    chosen_local_index = np.random.choice(num_valid, p=probabilities)
    chosen_action = actions[chosen_local_index]
    return chosen_action, probabilities[chosen_local_index]


def get_action_index(action: np.array):
    global actions
    # Find the index of the matching action in the actions array
    action_index = np.where(np.all(actions == action, axis=1))[0][0]
    return action_index

def test_normal_move():
    global track
    global actions

    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (3, 4)
    velocity = (0, 2)
    action = (0, 1)
    new_state, new_velocity, reward = step(
        state=state, 
        velocity=velocity, 
        action=action,
        actions=actions,
        track=track,
    )
    assert reward == -1
    assert np.array_equal(new_velocity, [0, 3])
    assert np.array_equal(new_state, [3, 7])

def test_out_of_bounds():
    global track 
    global actions

    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (3, 15)
    velocity = (0, 2)
    action = (0, 1)
    _, new_velocity, reward = step(
        state=state, 
        velocity=velocity, 
        action=action,
        actions=actions,
        track=track,
    )
    assert reward == -100
    assert np.array_equal(new_velocity, [0, 0])

def test_collision():
    global track 
    global actions
    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (2, 3)
    velocity = (0, -1)
    action = (0, 0)  # No acceleration, should hit wall
    _, _, reward = step(
        state=state, 
        velocity=velocity, 
        action=action,
        actions=actions,
        track=track,
    )
    assert reward == -100

def test_max_speed():
    global track 
    global actions
    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (2, 3)
    velocity = (4, 4)
    action = (2, 2)  
    new_state, new_velocity, reward = step(
        state=state, 
        velocity=velocity, 
        action=action,
        actions=actions,
        track=track,
    )
    assert reward == -1
    assert np.array_equal(new_velocity, [5, 5])
    assert np.array_equal(new_state, [7, 8])

def test_finish_line():
    global track 
    global actions
    _logger.debug("TRACK:")
    _logger.debug(track)
    _logger.debug(track[track.shape[0] - 1])
    state = (31, 11)
    velocity = (0, 5)
    action = (0, 1)  # No acceleration, should reach finish
    _, _, reward = step(
        state=state, 
        velocity=velocity, 
        action=action,
        actions=actions,
        track=track,
    )
    assert reward == 0

    
@pytest.mark.parametrize("epsilon, expected_actions", [
    (0.0, [-1, 0]),  # Greedy selection
    (1.0, 
        [
            [-1, 0],  # Decrease vertical speed (UP)
            [1, 0],   # Increase vertical speed (DOWN)
            [0, -1],  # Decrease horizontal speed (LEFT)
            [0, 1],   # Increase horizontal speed (RIGHT)
            [-1, -1], # Decrease both (UP LEFT)
            [-1, 1],  # Decrease vertical, increase horizontal (UP RIGHT)
            [1, -1],  # Increase vertical, decrease horizontal (DOWN LEFT)
            [1, 1],   # Increase both (DOWN RIGHT)
            [0, 0],   # No acceleration
        ]
    ),  # Random selection
])
def test_e_soft(epsilon, expected_actions):
    global actions
    # Sample Q-table
    Q = np.array([
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 8], 
            [9, 8, 7, 6, 5, 4, 3, 2, 1]
        ],  # Q-values for state (0,0) and (0,1)
        [
            [2, 2, 2, 2, 5, 4, 4, 2, 2], 
            [1, 0, 3, 3, 0, 1, 2, 2, 3]
        ]   # Q-values for state (1,0) and (1,1)
    ])

    _logger.debug(f"Shape of action value array: {Q.shape}")
    _logger.debug(f"Expected actions: {expected_actions}")
    state = (0, 1)
    np.random.seed(42)  # Fix seed for reproducibility
    action, action_prob = e_soft(
        epsilon=epsilon, 
        state=state, 
        Q=Q, 
        actions=actions,
    )
    _logger.debug(f"Produced action {action}")

    truth_values = np.array(action == expected_actions)
    _logger.debug(truth_values)

    # Convert action to tuple before checking
    assert np.any(action == expected_actions), f"Unexpected action {action} for ε={epsilon}"

def test_greedy():
    global actions
    # Sample Q-table
    Q = np.array([
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 8], 
            [9, 8, 7, 6, 5, 4, 3, 2, 1]
        ],  # Q-values for state (0,0) and (0,1)
        [
            [2, 2, 2, 2, 5, 4, 4, 2, 2], 
            [1, 0, 3, 3, 0, 1, 2, 2, 3]
        ]   # Q-values for state (1,0) and (1,1)
    ])
    state = (0, 1)
    action = greedy(
        state=state, 
        Q=Q, 
        actions=actions,
    )

    assert action[0] == -1 and action[1] == 0, f"Greedy function failed, expected [0, 0] or [1, 1], got {action}"

def test_greedy_tiebreak():
    global actions
    # Sample Q-table
    Q = np.array([
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 8], 
            [9, 8, 7, 6, 5, 4, 3, 2, 1]
        ],  # Q-values for state (0,0) and (0,1)
        [
            [2, 2, 2, 2, 2, 2, 1, 2, 2], 
            [1, 0, 3, 3, 0, 1, 2, 2, 3]
        ]   # Q-values for state (1,0) and (1,1)
    ])
    
    state = (1, 0)  # All Q-values are equal

    # Test with multiple seeds
    for seed in [42, 7, 123, 999, 2024]:
        np.random.seed(seed)  # Set random seed
        action = greedy(state, Q, actions=actions)
        
        assert not(action[0] == 1 and action[1] == -1), f"Greedy tiebreak failed with seed {seed}, got {action}"





import matplotlib.pyplot as plt

def plot_cumulative_timesteps(timesteps_per_episode, file_name="cumulative_timesteps.png"):
    """
    Plots the cumulative timesteps vs episodes.
    
    Parameters:
    - timesteps_per_episode: List[int] -> Number of timesteps for each episode.
    - file_name: str -> Name of the file to save the plot.
    """
    cumulative_timesteps = np.cumsum(timesteps_per_episode)  # Compute cumulative sum
    episodes = np.arange(1, len(timesteps_per_episode) + 1)  # Episode numbers

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_timesteps, episodes, color="red")
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.title("Episodes vs. Time Steps")
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()

def plot_trajectory(Q, actions, track, start_cells, file_name="trajectory_refined.png"):

    fig = plt.figure(figsize=(12, 8), dpi=150)
    fig.suptitle("Sample Trajectories from 10 Random Restarts using max 10000 timesteps", size=12, weight="bold")

    _logger.info("Creating trajectory based on the learned value function.")
    for i in range(10):
        _logger.info(f"Simulating trajectory {i}")
        # Reset to random starting position
        state = start_cells[np.random.choice(len(start_cells))]
        velocity = np.array([0, 0])

        # Store trajectory for this random restart
        random_trajectory = [state.tolist()]
        steps = 0

        while track[state[0], state[1]] != 2 and steps < 10000:  # Simulate up to 10000 steps
            action, _ = greedy(q_values=Q[state[0], state[1], :], actions=actions)
            state, velocity, _ = step(
                state=state, 
                velocity=velocity, 
                action=action, 
                track=track,
                start_cells=start_cells,
            )

            random_trajectory.append(state.tolist())  # Append state to trajectory

            steps += 1

        # Plot the trajectory on the corresponding subplot
        ax = plt.subplot(2, 5, i + 1)
        ax.axis("off")
        ax.imshow(track, cmap="PuBuGn", origin="upper")
        ax.plot(
            [pos[1] for pos in random_trajectory],
            [pos[0] for pos in random_trajectory],
            color="yellow", linewidth=1.5, alpha=0.8, label="Trajectory"
        )
        ax.scatter(
            random_trajectory[0][1], random_trajectory[0][0],
            color="green", label="Start", s=20
        )
        ax.scatter(
            random_trajectory[-1][1], random_trajectory[-1][0],
            color="red", label="End", s=20
        )
        ax.set_title(f"Restart {i + 1} with steps: {steps}", fontsize=10)

    plt.tight_layout()
    plt.legend(loc="upper left", fontsize=6)
    plt.savefig(file_name)
    plt.show()






if __name__ == "__main__":

    def parse_lists(value):
        try:
            return [int(x) for x in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError("Lists must be in the format 'x,y' (e.g., '3,1').")

    parser = argparse.ArgumentParser(description="Calculate the optimal policy for a nxm windy Gridworld.")

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Max Episodes. Default is 500.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Epsilon. Default is 0.1.",
    )

    parser.add_argument(
        "--discount-factor",
        type=float,
        default=1.0,
        help="Discount factor. Default is 1.",
    )

    args = parser.parse_args()

    #pytest.main()

    run(args=args)