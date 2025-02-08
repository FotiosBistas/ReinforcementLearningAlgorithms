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
MAX_SPEED = 5

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

# Define start and finish line positions
start_cols = list(range(3, 9))  # Columns 4 to 9
fin_cells = {
    (26, cols-1),
    (27, cols-1),
    (28, cols-1),
    (29, cols-1),
    (30, cols-1),
    (31, cols-1),
}  # Finish line cells

# Convert finish line positions to track representation
for r, c in fin_cells:
    track[r, c] = 2  # Mark finish cells with 2

# Print the track for debugging
_logger.debug(track)


def step(
    state: Tuple[int, int], 
    velocity: Tuple[int, int], 
    action: Tuple[int,int], 
    actions: List[np.array],
    track: np.array,
) -> Tuple[Tuple[int, int], Tuple[int, int], int]:

    # Convert inputs to NumPy arrays for easy operations
    state = np.array(state)
    velocity = np.array(velocity)
    action = np.array(action)

    # Update velocity with acceleration
    new_velocity = velocity + action
    new_velocity = np.clip(new_velocity, -MAX_SPEED, MAX_SPEED)

    _logger.debug(f"Current state: {state}")
    _logger.debug(f"Updated velocity from: {velocity} to {new_velocity}")

    # Update position with new velocity
    new_state = state + new_velocity

    # **Return immediately if out of bounds**
    if not (0 <= new_state[0] < track.shape[0]) or not (0 <= new_state[1] < track.shape[1]):
        new_start = np.array([0, np.random.choice(a=start_cols, size=1)[0]])  # Reset to start line
        new_velocity[:] = 0
        _logger.debug(f"OUT OF BOUNDS DETECTED: {new_state}. Resetting to start position {new_start}.")
        return new_start, new_velocity, -100

    # **Check for collision with a wall**
    if track[new_state[0], new_state[1]] == 1:
        new_start = np.array([0, np.random.choice(a=start_cols, size=1)[0]])  # Reset to start line
        new_velocity[:] = 0 
        _logger.debug(f"COLLISION detected at {new_state}. Resetting to start position {new_start}.")
        return new_start, new_velocity, -100  # Large negative reward

    # **Check if the car has reached the finish line**
    if track[new_state[0], new_state[1]] == 2:
        _logger.debug("FINISH LINE REACHED!")
        return new_state, new_velocity, 0  # Reached finish line

    # **Normal step with small penalty for each move**
    _logger.debug(f"VALID MOVE: Moving to {new_state} from {state}. Continuing the race.")
    return new_state, new_velocity, -1

def e_soft(
    epsilon: float, 
    state: Tuple[int, int],
    Q: np.ndarray,
    actions: np.ndarray
):
    """
    ε-soft policy: Returns an action and its probability of being chosen.
    """
    probabilities = np.ones(len(actions)) * (epsilon / len(actions))  # Exploration probabilities
    best_action_index = np.argmax(Q[state[0], state[1], :])  # Greedy action index
    probabilities[best_action_index] += 1 - epsilon  # Assign higher probability to greedy action

    # Sample action according to probabilities
    action_index = np.random.choice(len(actions), p=probabilities)
    action = actions[action_index]
    return action, probabilities[action_index]

def greedy(
    state: Tuple[int, int],
    Q: np.ndarray, 
    actions: np.ndarray
):
    """
    Greedy policy: Returns the action with the highest Q-value.
    """
    # Exploitation: Choose the best action(s)
    current_action_values_ = Q[state[0], state[1], :]
    best_action_index = np.argmax(current_action_values_)  # Single max computation
    action = actions[best_action_index]
    _logger.debug(f"Greedy action for state {state}: {action}")
    return action


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


def create_episode(
    epsilon: float,
    Q: np.array,
    track: np.array, 
    actions: List[np.array],
    start_cols: List[int],
    behavior_policy: Callable,
):

    episode_states = []
    episode_actions = []
    episode_action_probability = []
    episode_rewards = []

    episode_velocity = np.array([0, 0])

    # Initial states are in the start of the Track array
    initial_state = np.array([0, np.random.choice(a=start_cols, size=1)[0]])
    _logger.debug(f"Starting from state: {initial_state}")
    initial_action, action_prob = behavior_policy(
        epsilon=epsilon, 
        state=initial_state, 
        Q=Q, 
        actions=actions,
    )

    episode_states.append(initial_state)
    episode_actions.append(initial_action)
    episode_action_probability.append(action_prob)

    state, episode_velocity, reward = step(
        state=initial_state,
        velocity=episode_velocity, 
        action=initial_action,
        actions=actions, 
        track=track,
    )

    while track[state[0], state[1]] != 2: 
        episode_rewards.append(reward)
        episode_states.append(state)
        action, action_prob = behavior_policy(
            epsilon=epsilon,
            state=state, 
            Q=Q,
            actions=actions,
        )
        episode_actions.append(action)
        episode_action_probability.append(action_prob)
        state, episode_velocity, reward = step(
            state=state,
            velocity=episode_velocity, 
            action=action,
            actions=actions, 
            track=track,
        )

    # Append the last reward of the Terminal State
    episode_rewards.append(reward)

    return episode_states, episode_actions, episode_rewards, episode_action_probability



def run(args): 

    global actions
    global track
    global start_cols

    max_episodes = args.max_episodes
    discount_factor = args.discount_factor
    epsilon = args.epsilon

    Q = np.zeros((rows, cols, len(actions)))
    Counts = np.zeros((rows, cols, (len(actions))))
    p_s = greedy
    b = e_soft

    for episode in trange(max_episodes):

        episode_states, episode_actions, episode_rewards, episode_action_probability = create_episode(
            epsilon=epsilon, 
            Q=Q, 
            actions=actions,
            track=track,
            start_cols=start_cols,
            behavior_policy=b,
        )

        _logger.info(f"Created episode successfully")

        G = 0 
        W = 1

        for state, action, reward, prob in zip(reversed(episode_states), reversed(episode_actions), reversed(episode_rewards), reversed(episode_action_probability)):
            G = discount_factor * G + reward
            Counts[state[0], state[1], get_action_index(action)] += W 
            Q[state[0], state[1], get_action_index(action)] += (
                W / Counts[state[0], state[1], get_action_index(action)]
            ) * (G - Q[state[0], state[1], get_action_index(action)])


            best_action = p_s(state=state, Q=Q, actions=actions)
            _logger.debug(f"Best action using greedy policy: {best_action}")
            _logger.debug(f"Current action: {action}")
            if np.array_equal(action, best_action): 
                _logger.info("Breaking from inner loop. Current action is the best action.")
                break

            # 9 actions with equal probability by the behavior policy
            W *= prob

    plot_q_value_convergence(Q)  # Show learned Q-values
    plot_policy(Q, actions)  # Show best actions
    #episode_states, _, _, _ = create_episode(epsilon=epsilon, Q=Q, behavior_policy=e_soft, start_cols=start_cols, track=track, actions=actions)
    #plot_agent_path(episode_states)  # Show agent's movement


import matplotlib.pyplot as plt

def plot_q_value_convergence(Q):
    """
    Plots the maximum Q-value for each state over time.
    """
    q_values = np.max(Q, axis=2)  # Get max Q-value for each state
    plt.figure(figsize=(10, 5))
    plt.imshow(q_values, cmap="coolwarm", origin="upper")
    plt.colorbar(label="Max Q-value")
    plt.title("Q-value Heatmap for Each State")
    plt.xlabel("Track Column")
    plt.ylabel("Track Row")
    plt.savefig("q_value_convergence.png")

def plot_policy(Q, actions):
    """
    Plots the optimal policy as arrows on the racetrack.
    """
    policy_grid = np.zeros((Q.shape[0], Q.shape[1], 2))  # Stores best action vectors

    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if track[i, j] == 1:  # Wall
                continue
            best_action_index = np.argmax(Q[i, j, :])  # Best action index
            policy_grid[i, j] = actions[best_action_index]  # Store action vector

    plt.figure(figsize=(10, 5))
    plt.imshow(track, cmap="gray", origin="upper")  # Show track
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if track[i, j] == 1:
                continue
            plt.arrow(j, i, policy_grid[i, j, 1] * 0.4, -policy_grid[i, j, 0] * 0.4,
                      head_width=0.3, head_length=0.3, fc='red', ec='red')

    plt.title("Learned Policy (Best Actions)")
    plt.xlabel("Track Column")
    plt.ylabel("Track Row")
    plt.savefig("optimal_policy.png")

def plot_agent_path(episode_states):
    """
    Plots the agent's path on the racetrack.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(track, cmap="gray", origin="upper")  # Show track
    
    # Extract x and y positions from states
    x_positions = [s[1] for s in episode_states]
    y_positions = [s[0] for s in episode_states]

    plt.plot(x_positions, y_positions, marker="o", markersize=4, color="blue", label="Agent Path")
    plt.scatter(x_positions[0], y_positions[0], color="green", s=100, label="Start")
    plt.scatter(x_positions[-1], y_positions[-1], color="red", s=100, label="End")

    plt.title("Agent Path on Racetrack")
    plt.xlabel("Track Column")
    plt.ylabel("Track Row")
    plt.legend()
    plt.savefig("agent_path.png")



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

    pytest.main()

    run(args=args)