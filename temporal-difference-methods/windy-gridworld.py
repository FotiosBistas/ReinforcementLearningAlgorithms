import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from typing import Tuple, List

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("INFO")

actions = np.array([
    #UP
    np.array([-1, 0]),
    #DOWN
    np.array([1, 0]),
    #LEFT
    np.array([0, -1]),
    #RIGHT
    np.array([0, 1]),
])


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

def choose_action(
    epsilon: float,
    state: np.array,
    Q: np.array, 
):
    global actions
    # Choose action using policy derived from Q
    if np.random.random() < epsilon: 
        action_index = np.random.choice(a=len(actions))
        action = actions[action_index]
        _logger.debug(f"Randomly chose action: {action}")
    else:
        # Exploitation: Choose the best action(s), including handling of zero Q-values
        current_action_values_ = Q[state[0], state[1], :]
        _logger.debug(f"Current action values in state:\nAction Values: {current_action_values_}\nState: {state}")

        # Find indices of actions with the maximum Q-value (handles zero-Q values naturally)
        best_action_indices = np.flatnonzero(current_action_values_ == np.max(current_action_values_))

        # Randomly choose among the best actions
        action_index = np.random.choice(best_action_indices)
        action = actions[action_index]
        _logger.debug(f"Chose best action: {action}")

    return action

def get_action_index(action: np.array):
    global actions
    # Find the index of the matching action in the actions array
    action_index = np.where(np.all(actions == action, axis=1))[0][0]
    return action_index

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
    max_episodes = args.max_episodes
    _logger.info(f"Max Episodes: {max_episodes}")
    start = args.start
    _logger.info(f"Start: {start}")
    goal = args.goal
    _logger.info(f"Goal: {goal}")
    reward = args.reward 
    _logger.info(f"Reward: {reward}")

    time_steps = []        
    episode_durations = [] 

    cumulative_steps = 0 

    for episode in trange(max_episodes):
        _logger.debug(f"Running episode: {episode}")
        state = np.array(start)
        _logger.debug(f"Start state: {state}")

        action = choose_action(
            epsilon=epsilon,
            state=state, 
            Q=Q,
        )

        steps_in_episode = 0  

        while not np.array_equal(state, goal): 
            # keep previous state for Sarsa Update
            next_state = step(
                state=state, 
                action=action,
                n=n,
                m=m,
                wind=wind_column,
            )

            next_action = choose_action(
                epsilon=epsilon,
                state=next_state,
                Q=Q,
            )

            Q[state[0], state[1], get_action_index(action=action)] += alpha * (
                reward 
                + Q[next_state[0], next_state[1], get_action_index(next_action)] 
                - Q[state[0], state[1], get_action_index(action)]
            )

            action = next_action
            state = next_state
            steps_in_episode += 1

        cumulative_steps += steps_in_episode
        time_steps.append(cumulative_steps) 
        episode_durations.append(steps_in_episode)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, range(max_episodes), color="red", linewidth=1)
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    plt.title("Episodes vs. Time Steps")
    plt.grid()
    plt.savefig("./timesteps.png")

    plt.figure(figsize=(10, 6))
    plt.plot(range(max_episodes), episode_durations, label="Episode Duration", color="blue")
    plt.axhline(y=max(episode_durations), color="red", linestyle="--", label="Max Duration")
    plt.axhline(y=min(episode_durations), color="green", linestyle="--", label="Min Duration")
    plt.xlabel("Episodes")
    plt.ylabel("Episode Duration")
    plt.title("Episode Durations Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("./episode_durations.png")


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
    

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        help="Max Episodes. Default is 500.",
    )

    args = parser.parse_args()

    run(args=args)