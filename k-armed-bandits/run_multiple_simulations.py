import numpy as np
import subprocess
import argparse
import matplotlib.pyplot as plt
import os
import json
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG,
    format='%(filename)s - %(asctime)s - %(levelname)s - %(levelname)s - %(message)s'
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


def main(args):
    results = {}
    
    # Make sure the output directory exists
    os.makedirs("./output", exist_ok=True)

    for strategy in args.strategy:
        results[strategy] = {}

        _logger.info(f"Running with strategy {strategy}")

        for update_rule in args.update_rules:
            results[strategy][update_rule] = {}

            if strategy == "epsilon-greedy":
                for epsilon in args.epsilons:
                    for optimistic_initial_value in args.optimistic_initial_values:
                        _logger.info(f"Running {strategy} with epsilon = {epsilon}, optimistic initial value = {optimistic_initial_value}, and update rule = {update_rule}")
                        results[strategy][update_rule][(epsilon, optimistic_initial_value)] = run_single_simulation(
                            args, script_name="./_greedy.py", strategy=strategy, update_rule=update_rule, epsilon=epsilon, optimistic_initial_value=optimistic_initial_value
                        )
            elif strategy == "upper-confidence-bound":
                for c in args.cs:
                    _logger.info(f"Running {strategy} with c = {c} and update rule = {update_rule}")
                    results[strategy][update_rule][c] = run_single_simulation(
                        args, script_name="./_upper_confidence_bound.py", strategy=strategy, update_rule=update_rule, c=c
                    )
            elif strategy == "soft-max-preference":
                for gradient_alpha in args.gradient_alphas:
                    for use_baseline in args.use_baselines: 
                        _logger.info(f"Running {strategy} with gradient_alpha = {gradient_alpha} and use baseline = {use_baseline}")
                        results[strategy][update_rule][(gradient_alpha, use_baseline)] = run_single_simulation(
                            args, script_name="./_soft_max_preference.py", strategy=strategy, update_rule=update_rule, gradient_alpha=gradient_alpha, use_baseline=use_baseline
                        )

    # Plotting results for all strategies and update rules
    plt.figure(figsize=(10, 6))
    for strategy, strategy_results in results.items():
        for update_rule, update_results in strategy_results.items():
            for param, result in update_results.items():
                average_reward = np.array(result["average_reward"])
                _logger.debug(f"Shape of average reward array: {average_reward.shape}")
                if strategy == "epsilon-greedy":
                    label = f'{strategy} ε = {param[0]}, optimistic value = {param[1]}, {update_rule}'
                elif strategy == "upper-confidence-bound":
                    label = f'{strategy} c = {param}, {update_rule}'
                elif strategy == "soft-max-preference":
                    label = f'{strategy} α = {param[0]}, baseline = {param[1]}'
                plt.plot(average_reward, label=label)

    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time for Multiple Strategies, Update Rules, & Parameters')
    plt.grid(True)
    plt.legend()
    plt.savefig("./output/combined_average_reward_strategies.png")
    plt.show()

    # Plot Optimal Action Percentage if available
    plt.figure(figsize=(10, 6))
    for strategy, strategy_results in results.items():
        for update_rule, update_results in strategy_results.items():
            for param, result in update_results.items():
                optimal_action_percentage = np.array(result["optimal_action_percentage"])  # Check for optimal action percentage
                _logger.debug(f"Shape of optimal action percentage array: {optimal_action_percentage.shape}")
                if strategy == "epsilon-greedy":
                    label = f'{strategy} ε = {param[0]}, optimistic value = {param[1]}, {update_rule}'
                elif strategy == "upper-confidence-bound":
                    label = f'{strategy} c = {param}, {update_rule}'
                elif strategy == "soft-max-preference":
                    label = f'{strategy} α = {param[0]}, baseline = {param[1]}'
                plt.plot(optimal_action_percentage, label=f'Optimal Action {label}')
    plt.xlabel('Iterations')
    plt.ylabel('% Optimal Action Selected')
    plt.title('Optimal Action Selection Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig("./output/combined_optimal_action_percentage.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multiple Multi-Armed Bandit Simulations")

    # Environment parameters
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')
    # Add a command-line argument for logging level
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
    parser.add_argument('--gradient-alphas', nargs="+", type=float, default=[0.1, 0.01], help='List of gradient alphas for softmax-preferences')
    parser.add_argument('--optimistic-initial-values', nargs="+", type=int, default=[0,10], help="List of optimistic initial values to use for epsilon greedy.")
    parser.add_argument('--use-baselines', nargs="+", type=int, default=[0, 1], help="List of use baselines options for softmax-preferences.")


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

    # Formatting parameters
    parser.add_argument('--output-plot', type=int, choices=[0, 1], help='Output the plots')

    # Save results to disk
    parser.add_argument('--save-to-disk', type=int, choices=[0, 1], help="Saves the program results to the disk")
    
    args = parser.parse_args()

    # Set up logging with the level specified in the command line
    logger = setup_logging(LOGGING_LEVELS[args.log_level])

    main(args)
