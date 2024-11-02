import os 
import json
import subprocess
import logging
import argparse

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

def run_single_simulation(args, script_name: str, strategy: str, update_rule: str, optimistic_initial_value: float=None, gradient_alpha: float=None, epsilon: float = None, c: float = None, use_baseline: int=None):
    command = [
        "python", str(script_name),  
        "--seed", str(args.seed),
        "--iterations", str(args.iterations),
        "--runs", str(args.runs),
        "--log-level", str(args.log_level),
        "--env-type", str(args.env_type),
        "--reward-std", str(args.reward_std),
        "--random-walk-step", str(args.random_walk_step),
        "--update-rule", str(update_rule),
        "--alpha", str(args.alpha),
        "--save-to-disk", str(1), 
        "--output-plot", str(1),  
    ]

    # Strategy-specific arguments
    if strategy == "epsilon-greedy":
        if epsilon is not None:
            command.extend(["--epsilon", str(epsilon)])  # Add epsilon if using epsilon-greedy
        if optimistic_initial_value is not None:
            command.extend(["--optimistic-initial-values", str(optimistic_initial_value)])  # Add optimistic initial value
    elif strategy == "upper-confidence-bound" and c is not None:
        command.extend(["--c", str(c)])  # Add c for UCB strategy
    elif strategy == "soft-max-preference" and gradient_alpha is not None:
        command.extend(["--gradient-alpha", str(gradient_alpha)])  # Add gradient_alpha for softmax-preferences strategy
        command.extend(["--use-baseline", str(use_baseline)])  # Add baseline option for softmax-preferences

    # Create log and result file names based on strategy, parameter, update rule, optimistic initial value, and baseline reward setting
    if strategy == "epsilon-greedy" and epsilon is not None:
        log_file = f"./output/epsilon_{epsilon}_optimistic_{optimistic_initial_value}_update_{update_rule}_log.txt"
        result_file = f"./output/epsilon_{epsilon}_optimistic_{optimistic_initial_value}_update_{update_rule}_results.json"
    elif strategy == "upper-confidence-bound" and c is not None:
        log_file = f"./output/ucb_{c}_update_{update_rule}_log.txt"
        result_file = f"./output/ucb_{c}_update_{update_rule}_results.json"
    elif strategy == "soft-max-preference" and gradient_alpha is not None:
        log_file = f"./output/softmax_{gradient_alpha}_usebaseline_{use_baseline}_log.txt"
        result_file = f"./output/softmax_{gradient_alpha}_usebaseline_{use_baseline}_results.json"

    _logger.info(f"Running command {command}")
    # Run the simulation and write output to a log file
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=f, text=True)
    
    # Read the results file
    with open(result_file, "r") as f:
        return json.load(f)  # Load the results of the run

def run_simulations_and_plot(args):
    results = {}

    # Make sure the output directory exists
    os.makedirs("./output", exist_ok=True)

    parameter_values = []
    average_rewards_egreedy = []
    average_rewards_ucb = []
    average_rewards_gradient = []
    average_rewards_greedy_opt = []

    # ε-greedy for different ε values
    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]  
    for epsilon in epsilons:
        result = run_single_simulation(
            args, script_name="./_greedy.py", strategy="epsilon-greedy", update_rule="mean", epsilon=epsilon, optimistic_initial_value=0.0
        )
        average_reward = np.mean(result["average_reward"][:1000])  # Average over the first 1000 steps
        average_rewards_egreedy.append(average_reward)
    parameter_values.append(epsilons)

    ## UCB for different c values
    cs = [1/16, 1/4, 1/2, 1.0, 2.0, 4.0]
    for c in cs:
        result = run_single_simulation(
            args, script_name="./_upper_confidence_bound.py", strategy="upper-confidence-bound", update_rule="mean", c=c
        )
        average_reward = np.mean(result["average_reward"][:1000])  # Average over the first 1000 steps
        average_rewards_ucb.append(average_reward)
    parameter_values.append(cs)

    ## Gradient Bandit for different alpha values
    gradient_alphas = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0]
    for alpha in gradient_alphas:
        result = run_single_simulation(
            args, script_name="./_soft_max_preference.py", strategy="soft-max-preference", update_rule="mean", gradient_alpha=alpha, use_baseline=1
        )
        average_reward = np.mean(result["average_reward"][:1000])  # Average over the first 1000 steps
        average_rewards_gradient.append(average_reward)
    parameter_values.append(gradient_alphas)

    # Greedy with optimistic initialization for different Q0 values
    q0_values = [1/4, 1/2, 1.0, 2.0, 4.0]
    #q0_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
    for q0 in q0_values:
        result = run_single_simulation(
            args, script_name="./_greedy.py", strategy="epsilon-greedy", update_rule="exponential-decay", epsilon=0.0, optimistic_initial_value=q0
        )
        average_reward = np.mean(result["average_reward"][:1000])  # Average over the first 1000 steps
        average_rewards_greedy_opt.append(average_reward)
    parameter_values.append(q0_values)

    # Plotting results for all strategies with parameters on the x-axis
    plt.figure(figsize=(10, 6))

    # ε-greedy plot
    plt.plot(epsilons, average_rewards_egreedy, label='ε-greedy', color='red', marker='o')

    ## UCB plot
    plt.plot(cs, average_rewards_ucb, label='UCB', color='blue', marker='o')

    ### Gradient Bandit plot
    plt.plot(gradient_alphas, average_rewards_gradient, label='Gradient Bandit', color='green', marker='o')

    # Greedy with optimistic initialization plot
    plt.plot(q0_values, average_rewards_greedy_opt, label='Greedy with optimistic initialization, α = 0.1', color='black', marker='o')

    plt.xscale('log') 
    # Define the custom tick values and labels for the x-axis
    ticks = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    tick_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    plt.xticks(ticks, tick_labels)
    plt.xlabel('ε/ α/ c/ Q0')
    plt.ylabel('Average Reward over first 1000 steps')
    plt.title('Average Reward for Different Strategies and Parameters')
    plt.grid(True)

    # Add manual positioning of labels near curves as shown in the original image
    plt.text(1/4, 1.5, 'UCB', color='blue', fontsize=12)
    plt.text(1/4, 1.3, 'ε-greedy', color='red', fontsize=12)
    plt.text(1/4, 1.25, 'Gradient Bandit', color='green', fontsize=12)
    plt.text(1, 1.45, 'Greedy with optimistic initialization', color='black', fontsize=12)

    plt.legend()
    plt.savefig("./output/combined_average_reward_param_xaxis.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multiple Multi-Armed Bandit Simulations")

    # Environment parameters
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info', help='Set the logging level')
    parser.add_argument('--env-type', choices=['stationary', 'nonstationary'], default='stationary',
                        help='Type of environment: stationary or nonstationary')
    parser.add_argument('--random-walk-step', type=float, default=0.1,
                        help='Step size for random walk in nonstationary environments')
    parser.add_argument('--reward-std', type=float, default=1.0,
                        help='Standard deviation of the reward distribution')
    parser.add_argument('--runs', type=int, default=2000, help='Number of runs (independent simulations)')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations per run')
    parser.add_argument('--k', type=int, default=10, help='Number of bandits.')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha step size inside the exponential decay')

    # Strategy parameters
    parser.add_argument('--strategy', nargs='+', choices=['epsilon-greedy', 'upper-confidence-bound', 'soft-max-preference'],
                        required=True, help='Strategy to use in the simulation')
    parser.add_argument('--update-rules', nargs='+', choices=['mean', 'exponential-decay'], default=['mean'],
                        help='Update rules for the value estimates')
    # Epsilon values for epsilon-greedy strategy
    parser.add_argument('--epsilons', nargs='+', type=float, default=[0.1, 0.01, 0.0], 
                        help='List of epsilon values to use in epsilon-greedy strategy')

    # C values for UCB strategy
    parser.add_argument('--cs', nargs='+', type=float, default=[1.0, 2.0], help='List of c values to use in UCB strategy')

    # Gradient alpha values for softmax-preferences strategy
    parser.add_argument('--gradient-alphas', nargs='+', type=float, default=[0.1, 0.2], help='List of gradient alphas')

    # Optimistic initial values for epsilon-greedy
    parser.add_argument('--optimistic-initial-values', nargs='+', type=float, default=[0.1], help='Optimistic initial values')

    args = parser.parse_args()

    run_simulations_and_plot(args)
