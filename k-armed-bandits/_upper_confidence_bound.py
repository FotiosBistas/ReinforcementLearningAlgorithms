import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG,
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

_logger = logging.getLogger()

# Define a dictionary to map string level names to logging module constants
LOGGING_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL,
}

def setup_logging(level):
    """Set up logging configuration based on the provided logging level."""
    global _logger
    _logger.setLevel(level)

def get_reward(reward_std: float, action_value: float) -> float:
    _logger.debug(f"Getting reward for action value {action_value} using std: {reward_std}")
    return np.random.normal(loc=action_value, scale=reward_std)

def select_action(k:int, t: int, c: float, estimated_action_values: np.ndarray, action_counts: np.ndarray) -> int:
        ucb_values = np.zeros(k)
        for i in range(k): 
            # 1e-5 in the case of zero t=0 or action_counts[i] = 0
            ucb_values[i] = estimated_action_values[i] + c * np.sqrt(
                np.log(t + 1e-5) / (action_counts[i] + 1e-5)
            )
        return np.argmax(ucb_values)

def random_walk(action_values: np.ndarray, random_walk_step: float, k: int) -> None:
    #TODO is this based on the action_value?
    #TODO I don't think so
    _logger.debug(f"Performing random walk on the action values: {action_values}, using walk step: {random_walk_step}")
    action_values += np.random.normal(0, random_walk_step, size=k)
    _logger.debug(f"Action values after performing a random walk step: {action_values}")

def executor(args): 
    _logger.info(f"Running simulation with c: {args.c}")
    
    np.random.seed(seed=args.seed)
    rewards_aggregate = []
    optimal_action_selected_aggregate = []


    for run in range(args.runs): 
        rewards_per_run = []
        optimal_action_selected = []
        action_values = np.random.normal(0, 1, size=args.k)
        action_counts = np.zeros(args.k) 
        sum_rewards = np.zeros(args.k)
        estimated_action_values = np.full(args.k, args.optimistic_initial_values, dtype=float)

        for timestep in range(args.iterations):
            # Identify the optimal action (action with the highest true value)
            optimal_action = np.argmax(action_values)
            action = select_action(c=args.c, k=args.k, t=timestep, estimated_action_values=estimated_action_values, action_counts=action_counts) 
            reward = get_reward(reward_std=args.reward_std, action_value=action_values[action])
            rewards_per_run.append(reward)

            # Track if the selected action matches the optimal action
            if action == optimal_action:
                optimal_action_selected.append(1)
            else:
                optimal_action_selected.append(0)

            if args.update_rule == "mean": 
                action_counts[action] += 1
                sum_rewards[action] += reward
                estimated_action_values[action] = sum_rewards[action] / action_counts[action]
            elif args.update_rule == "exponential-decay":
                estimated_action_values[action] += args.alpha * (reward - estimated_action_values[action])

            if args.env_type == "nonstationary": 
                random_walk(action_values=action_values, k=args.k, random_walk_step=args.random_walk_step)
        
        rewards_aggregate.append(rewards_per_run)
        optimal_action_selected_aggregate.append(optimal_action_selected)

    # Calculate the average reward and optimal action selection rate over runs
    average_rewards = np.mean(np.vstack(rewards_aggregate), axis=0)
    optimal_action_percentage = np.mean(np.vstack(optimal_action_selected_aggregate), axis=0)
    
    return average_rewards, optimal_action_percentage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a greedy bandit")

    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info', help='Set the logging level')
    parser.add_argument('--env-type', choices=['stationary', 'nonstationary'], default='stationary',
                        help='Type of environment: stationary or nonstationary')
    parser.add_argument('--random-walk-step', type=float, default=0.1,
                        help='Step size for random walk in nonstationary environments')
    parser.add_argument('--reward-std', type=float, default=1.0,
                        help='Standard deviation of the reward distribution')
    parser.add_argument('--optimistic-initial-values', type=float, default=0)
    parser.add_argument('--runs', type=int, default=2000, help='Number of runs (independent simulations)')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations per run')
    parser.add_argument('--k', type=int, default=10, help='Number of bandits.')
    parser.add_argument('--update-rule', choices=['mean', 'exponential-decay'], default='mean',
                        help='Update rule for the value estimates')
    parser.add_argument('--c',  type=float, default=0.1,
                        help='C values for upper-confidence-bound strategy')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--save-to-disk', type=int, choices=[0, 1], help="Saves the program results to the disk")
    parser.add_argument('--output-plot', type=int, choices=[0, 1], help="Output the plots")

    args = parser.parse_args()

    setup_logging(LOGGING_LEVELS[args.log_level])

    average_rewards, optimal_action_percentage = executor(args)

    if args.output_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(average_rewards, label=f'c = {args.c}')
        plt.xlabel('Iterations')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Over Time for UCB Strategy')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.savefig(f"./output/average_reward_c_{args.c}.png")

        plt.figure(figsize=(10, 6))
        plt.plot(optimal_action_percentage * 100, label=f'Optimal Action % for c = {args.c}')
        plt.xlabel('Iterations')
        plt.ylabel('% Optimal Action Selected')
        plt.title('Optimal Action Selection Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.savefig(f"./output/optimal_action_percentage_c_{args.c}.png")

    if args.save_to_disk:
        result_file = f"./output/ucb_{args.c}_update_{args.update_rule}_results.json"
        with open(result_file, "w") as f:
            json.dump({
                'average_reward': list(average_rewards),
                'optimal_action_percentage': list(optimal_action_percentage)
            }, f)

