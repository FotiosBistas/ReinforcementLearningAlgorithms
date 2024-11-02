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

def random_walk(action_values: np.ndarray, random_walk_step: float, k: int) -> None:
    #TODO is this based on the action_value?
    #TODO I don't think so
    _logger.debug(f"Performing random walk on the action values: {action_values}, using walk step: {random_walk_step}")
    action_values += np.random.normal(0, random_walk_step, size=k)
    _logger.debug(f"Action values after performing a random walk step: {action_values}")

def executor(args): 
    """
    Executor function that runs the bandit simulations.
    """
    np.random.seed(seed=args.seed)
    rewards_aggregate = []
    optimal_action_selected_aggregate = []

    for run in range(args.runs): 
        rewards_per_run = []
        optimal_action_selected = []

        # Initialize action values and preferences
        #action_values = np.random.normal(4, 1, size=args.k)  # True values
        action_values = np.random.normal(0, 1, size=args.k)  # True values

        action_preferences = np.zeros(args.k)  # Preferences for softmax
        baseline_rewards = []  # To compute the average reward

        for timestep in range(args.iterations):
            # Identify the optimal action (the action with the highest true value)
            optimal_action = np.argmax(action_values)

            # Compute softmax probabilities based on current preferences
            # Subtract the max for numerical stability (to prevent overflow in exp)
            exp_preferences = np.exp(action_preferences - np.max(action_preferences))
            probabilities_timestep = exp_preferences / np.sum(exp_preferences)
            
            # Select an action according to the softmax probabilities
            action = np.random.choice(a=len(action_preferences), p=probabilities_timestep)
            reward = get_reward(reward_std=args.reward_std, action_value=action_values[action])

            # Append the current reward
            rewards_per_run.append(reward)
            baseline_rewards.append(reward)

            # Track if the selected action matches the optimal action
            if action == optimal_action:
                optimal_action_selected.append(1)
            else:
                optimal_action_selected.append(0)

            # Update baseline reward
            if args.use_baseline == 0:
                baseline_reward = 0
            else:
                # Calculate the average reward excluding the current timestep
                baseline_reward = np.mean(baseline_rewards[:-1]) if len(baseline_rewards) > 1 else baseline_rewards[0] 

            # Update preference for the selected action
            action_preferences[action] = action_preferences[action] + args.gradient_alpha * (reward - baseline_reward) * (1 - probabilities_timestep[action])

            # Update preferences for all other actions
            for a in range(len(action_preferences)): 
                if a != action: 
                    action_preferences[a] = action_preferences[a] - args.gradient_alpha * (reward - baseline_reward) * probabilities_timestep[a]

            # Apply a random walk to action values if nonstationary environment
            if args.env_type == "nonstationary": 
                random_walk(action_values=action_values, k=args.k, random_walk_step=args.random_walk_step)

        # Store results for this run
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
    parser.add_argument('--epsilon', type=float, default=[0.1],
                        help='Epsilon values for epsilon-greedy strategy')
    parser.add_argument('--use-baseline', type=int, choices=[0,1], help="Whether to use a zero baseline reward.")
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gradient-alpha', type=float, default=0.1)
    parser.add_argument('--save-to-disk', type=int, choices=[0, 1], help="Saves the program results to the disk")
    parser.add_argument('--output-plot', type=int, choices=[0, 1], help="Output the plots")

    args = parser.parse_args()

    setup_logging(LOGGING_LEVELS[args.log_level])

    plt.figure(figsize=(10, 6))

    average_rewards, optimal_action_percentage = executor(args)

    if args.output_plot: 
        plt.figure(figsize=(10, 6))
        plt.plot(average_rewards, label=f'Softmax Preferences')
        plt.xlabel('Iterations')
        plt.ylabel('Average Reward')
        plt.title('Average Reward Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.savefig(f"./output/average_reward_softmax_preferences_{args.gradient_alpha}_usebaseline_{args.use_baseline}.png")

        plt.figure(figsize=(10, 6))
        plt.plot(optimal_action_percentage * 100, label=f'Optimal Action %')
        plt.xlabel('Iterations')
        plt.ylabel('% Optimal Action Selected')
        plt.title('Optimal Action Selection Over Time')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.savefig(f"./output/optimal_action_percentage_softmax_preference_{args.gradient_alpha}_usebaseline_{args.use_baseline}.png")

    if args.save_to_disk:
        result_file = f"./output/softmax_{args.gradient_alpha}_usebaseline_{args.use_baseline}_results.json"

        with open(result_file, "w") as f:
            json.dump({
                'average_reward': list(average_rewards),
                'optimal_action_percentage': list(optimal_action_percentage)
            }, f)
