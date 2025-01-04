import argparse
import logging
import numpy as np
import matplotlib as plt


from typing import Tuple, List

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")

actions = [
    #LEFT
    np.array([0, -1]),
    #RIGHT
    np.array([0, 1]),
    #UP
    np.array([-1, 0]),
    #DOWN
    np.array([1, 0]),
]


def step(
    state: Tuple[int, int], 
    action: Tuple[int, int],
    n: int, 
    m: int,
    wind: List[int],
):
    global actions

    new_state = np.array(state) + action

    new_state[0] -= wind[state[1]] 

    _logger.debug(f"Performed a step on state: {state} with action {action}")
    _logger.debug(f"Wind value on {(state[0], state[1])} is {wind[state[1]]}")

    new_state[0] = max(0, min(new_state[0], n - 1))  # Clamp row index
    new_state[1] = max(0, min(new_state[1], m - 1))  # Clamp column index

    return new_state

def run(args):

    n = args.n
    _logger.info(f"Height: {n}")
    m = args.m
    _logger.info(f"Width: {m}")

    Q = np.zeros((n,m,4))

    alpha = args.alpha
    _logger.info(f"Alpha: {alpha}")
    epsilon = args.epsilon
    _logger.info(f"Epsilon: {epsilon}")
    wind_column = args.wind_column
    _logger.info(f"Wind Column:\n{wind_column}")

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the optimal policy for a nxm windy Gridworld.")

    parser.add_argument(
        "--n",
        type=int,
        default=7,
        help="The height of the gridworld. Default value is 7.",
    )

    parser.add_argument(
        "--m",
        type=int,
        default=10,
        help="The width of the gridworld. Default value is 10.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="The exploration probability of the e-greedy policy. Default value is 0.1.",
    )

    parser.add_argument(
        "--alpha",
        type=float, 
        default=0.5, 
        help="The step size of the Temporal Difference Update. Default value is 0.5.",
    )

    parser.add_argument(
        "--reward",
        type=float,
        default=-1.0, 
        help="The reward for each state. Default value is -1.0.",
    )


    parser.add_argument(
        "--wind-column",
        type=int,
        nargs="+",
        default=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
        help="The wind for each column of the Gridworld. Default is [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]",
    )

    parser.add_argument(
        "--start",
        type=int, 
        nargs="+",
        default=[3,0],
        help="Start location of the car. Default [3,0].",
    )

    parser.add_argument(
        "--goal",
        type=int, 
        nargs="+",
        default=[3,7],
        help="Terminal state. Default [3,7].",
    )

    args = parser.parse_args()

    run(args=args)