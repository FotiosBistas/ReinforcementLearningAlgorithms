import argparse
import logging
import numpy as np

from typing import Tuple, List

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")

import numpy as np
from typing import Tuple

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
    action: Tuple[int, int], 
    n: int, 
    m: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Updates position and velocity of the car in the racetrack.

    Args:
        state (Tuple[int, int]): Current position (row, col)
        velocity (Tuple[int, int]): Current velocity (vy, vx)
        action (Tuple[int, int]): Acceleration applied (dy, dx)
        n (int): Number of rows in the grid
        m (int): Number of columns in the grid

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: New position, new velocity
    """
    # Convert inputs to NumPy arrays for easy operations
    state = np.array(state)
    velocity = np.array(velocity)
    action = np.array(action)

    # Update velocity with acceleration
    new_velocity = velocity + action

    # Update position with velocity
    new_state = state + new_velocity

    # Clamp within grid boundaries
    new_state[0] = np.clip(new_state[0], 0, n - 1)
    new_state[1] = np.clip(new_state[1], 0, m - 1)

    return tuple(new_state), tuple(new_velocity)


def get_action_index(action: np.array):
    global actions
    # Find the index of the matching action in the actions array
    action_index = np.where(np.all(actions == action, axis=1))[0][0]
    return action_index


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
        "--start-line",
        type=int, 
        nargs="+",
        default=[3,0],
        help="Start location of the car. Default [3,0].",
    )

    parser.add_argument(
        "--finish-line",
        type=int, 
        nargs="+",
        default=[3,7],
        help="Terminal state. Default [3,7].",
    )
    

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        help="Max Episodes. Default is 500.",
    )

    args = parser.parse_args()

    run(args=args)