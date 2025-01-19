import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List
from tqdm import trange

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


def Sarsa(
    n: int, 
    m: int,
    max_episodes: int,
    epsilon: float,
    start: np.array,
    goal: np.array,
    alpha: float,
    reward: float, 
    cliff_reward: float,
    cliffs: List[List[int]],
) -> List[float]:

    Q = np.zeros((n, m, 4))
    episode_rewards = []  # Track total reward for each episode

    for episode in trange(max_episodes, desc="Running Sarsa"):
        _logger.debug(f"Running episode {episode}")
        state = start
        total_reward = 0  # Initialize total reward for this episode

        action = choose_action(
            epsilon=epsilon,
            state=state, 
            Q=Q,
        )

        while not np.array_equal(state, goal): 
            next_state, actual_reward = step(
                state=state, 
                action=action,
                n=n,
                m=m,
                start=start,
                reward=reward,
                cliff_reward=cliff_reward,
                cliffs=cliffs,
            )

            total_reward += actual_reward  # Accumulate reward for this episode

            next_action = choose_action(
                epsilon=epsilon,
                state=next_state, 
                Q=Q,
            )

            Q[state[0], state[1], get_action_index(action=action)] += alpha * (
                actual_reward
                + Q[next_state[0], next_state[1], get_action_index(next_action)] 
                - Q[state[0], state[1], get_action_index(action)]
            )

            action = next_action
            state = next_state

        episode_rewards.append(total_reward)  # Store total reward for this episode

    return episode_rewards

def get_action_index(action: np.array):
    global actions
    # Find the index of the matching action in the actions array
    action_index = np.where(np.all(actions == action, axis=1))[0][0]
    return action_index

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

        return action 

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

def step(
    state: Tuple[int, int], 
    action: Tuple[int, int],
    n: int, 
    m: int,
    start: np.array, 
    reward: float, 
    cliff_reward: float,
    cliffs: List[List[int]],
) -> Tuple[np.array, float]:

    global actions

    new_state = np.array(state) + action

    _logger.debug(f"Performed a step on state: {state} with action {action}")

    new_state[0] = max(0, min(new_state[0], n - 1))  # Clamp row index
    new_state[1] = max(0, min(new_state[1], m - 1))  # Clamp column index

    if any((new_state == cliff).all() for cliff in cliffs):
        _logger.debug(f"Fell into cliff {new_state} with reward {cliff_reward}! Returning to start!")
        return start, -100


    _logger.debug(f"Advancing to new state {new_state} with reward {reward}")
    return new_state, -1

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
    gamma = args.gamma
    _logger.info(f"Gamma: {gamma}")
    max_episodes = args.max_episodes
    _logger.info(f"Max Episodes: {max_episodes}")
    start = args.start
    _logger.info(f"Start: {start}")
    goal = args.goal
    _logger.info(f"Goal: {goal}")
    reward = args.reward 
    _logger.info(f"Reward: {reward}")
    cliff_reward = args.cliff_reward 
    _logger.info(f"Cliff Reward: {cliff_reward}")
    cliffs = args.cliffs
    _logger.info(f"Cliffs: {cliffs}")

    rewards_sarsa = Sarsa(
        n=n,
        m=m,
        epsilon=epsilon,
        start=start,
        goal=goal,
        reward=reward,
        cliff_reward=cliff_reward,
        cliffs=cliffs,
        alpha=alpha,
        max_episodes=max_episodes,
    )

    # Plot results
    plt.plot(rewards_sarsa, label="Sarsa", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.title("Sarsa Algorithm Performance")
    plt.ylim(-100, 0)  # Limit y-axis range
    plt.legend()
    plt.grid()
    plt.savefig("./here.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the Sarsa, Q-learning and Expected Sarsa for a nxm cliff walking problem.")

    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="The height of the world. Default value is 4.",
    )

    parser.add_argument(
        "--m",
        type=int,
        default=12,
        help="The width of the world. Default value is 12.",
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
        "--gamma",
        type=float,
        default=1,
        help="The gamma value (discount) for Q Learning and Expected Sarsa. Default values is 1."
    )

    parser.add_argument(
        "--reward",
        type=float,
        default=-1.0, 
        help="The reward for each state. Default value is -1.0.",
    )

    def parse_cliffs(value):
        try:
            return [int(x) for x in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError("Cliffs must be in the format 'x,y' (e.g., '3,1').")


    parser.add_argument(
        "--cliffs",
        type=parse_cliffs,
        nargs="+",
        default=[[3,1], [3,2], [3,3], [3,4], [3,5], [3,6], [3,7], [3,8], [3,9], [3,10]],
        help="Cliff positions inside the grid. Default value is [3,1], [3,2], [3,3], [3,4], [3,5], [3,6], [3,7], [3,8], [3,9], [3,10].",
    )

    parser.add_argument(
        "--cliff-reward",
        type=float,
        default=-100.0,
        help="The reward for falling off the cliff. Default value is -100.0.",
    )

    parser.add_argument(
        "--start",
        type=int, 
        nargs="+",
        default=[3,0],
        help="Start location. Default [3,0].",
    )

    parser.add_argument(
        "--goal",
        type=int, 
        nargs="+",
        default=[3,11],
        help="Terminal state. Default [3,11].",
    )
    
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        help="Max Episodes. Default is 500.",
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help="Max Episodes for each Run. Default is 50.",
    )

    args = parser.parse_args()

    run(args=args)