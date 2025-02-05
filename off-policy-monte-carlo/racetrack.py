import argparse
import pytest
import logging
import numpy as np

from typing import Tuple, List

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")

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
) -> Tuple[Tuple[int, int], Tuple[int, int], int]:

    global track
    global actions

    # Convert inputs to NumPy arrays for easy operations
    state = np.array(state)
    velocity = np.array(velocity)
    action = np.array(action)

    # Update velocity with acceleration
    new_velocity = velocity + action
    _logger.debug(new_velocity)
    new_velocity = np.clip(new_velocity, -MAX_SPEED, MAX_SPEED)


    # Update position with new velocity
    new_state = state + new_velocity

    # **Return immediately if out of bounds**
    if not (0 <= new_state[0] < track.shape[0]) or not (0 <= new_state[1] < track.shape[1]):
        new_state = np.random.choice(start_cols)
        _logger.debug(f"OUT OF BOUNDS DETECTED: {new_state}. Resetting to start position {new_state}.")
        return (0, new_state), (0, 0), -100

    # **Check for collision with a wall**
    if track[new_state[0], new_state[1]] == 1:
        _logger.debug(f"COLLISION detected at {new_state}. Resetting to start position {new_state}.")
        new_state = (0, np.random.choice(start_cols))  # Reset to start line
        new_velocity = (0, 0)
        return new_state, new_velocity, -100  # Large negative reward

    # **Check if the car has reached the finish line**
    if track[new_state[0], new_state[1]] == 2:
        _logger.debug("FINISH LINE REACHED!")
        return tuple(new_state), tuple(new_velocity), 0  # Reached finish line

    # **Normal step with small penalty for each move**
    _logger.debug(f"VALID MOVE: Moving to {new_state}. Continuing the race.")
    return tuple(new_state), tuple(new_velocity), -1



def get_action_index(action: np.array):
    global actions
    # Find the index of the matching action in the actions array
    action_index = np.where(np.all(actions == action, axis=1))[0][0]
    return action_index

def test_normal_move():
    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (3, 4)
    velocity = (0, 2)
    action = (0, 1)
    new_state, new_velocity, reward = step(state, velocity, action)
    assert reward == -1
    assert new_velocity == (0, 3)
    assert new_state == (3, 7)

def test_out_of_bounds():
    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (3, 15)
    velocity = (0, 2)
    action = (0, 1)
    _, new_velocity, reward = step(state, velocity, action)
    assert reward == -100
    assert new_velocity == (0, 0)

def test_collision():
    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (2, 3)
    velocity = (0, -1)
    action = (0, 0)  # No acceleration, should hit wall
    new_state, new_velocity, reward = step(state, velocity, action)
    assert reward == -100

def test_max_speed():
    _logger.debug("TRACK:")
    _logger.debug(track)
    state = (2, 3)
    velocity = (4, 4)
    action = (2, 2)  
    new_state, new_velocity, reward = step(state, velocity, action)
    assert reward == -1
    assert new_velocity == (5, 5)
    assert new_state == (7, 8)

def test_finish_line():
    _logger.debug("TRACK:")
    _logger.debug(track)
    _logger.debug(track[track.shape[0] - 1])
    state = (31, 11)
    velocity = (0, 5)
    action = (0, 1)  # No acceleration, should reach finish
    new_state, new_velocity, reward = step(state, velocity, action)
    assert reward == 0



def run(args): 

    vertical_velocity = 0
    horizontal_velocity = 0

if __name__ == "__main__":

    def parse_lists(value):
        try:
            return [int(x) for x in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError("Lists must be in the format 'x,y' (e.g., '3,1').")

    parser = argparse.ArgumentParser(description="Calculate the optimal policy for a nxm windy Gridworld.")

    parser.add_argument(
        "--max-speed",
        type=float,
        default=5.0,
        help="Max Speed. Default is 5.0",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        help="Max Episodes. Default is 500.",
    )

    args = parser.parse_args()

    pytest.main()

    run(args=args)