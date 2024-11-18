import argparse
import logging
import numpy as np

from typing import Tuple
from pprint import pprint
from scipy.stats import poisson


logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")

def poisson_distribution(n: int, lambda_: float):
    return poisson.pmf(k=n, mu=lambda_)

def precompute_poisson_probabilities(max_cars: int, rates: Tuple[int, int]):
    probs_first = [poisson_distribution(n, rates[0]) for n in range(max_cars + 1)]
    probs_second = [poisson_distribution(n, rates[1]) for n in range(max_cars + 1)]
    return np.array(probs_first), np.array(probs_second)

def truncated_range(probabilities, threshold=1e-6):
    return [i for i, p in enumerate(probabilities) if p.astype(float) > threshold]



def policy_evaluation(
    state: Tuple[int, int],
    policy: int,
    value_function: np.ndarray,
    transfer_cost: float,
    max_cars: Tuple[int, int],
    return_rates: Tuple[int, int], 
    request_rates: Tuple[int, int], 
    discount_rate: float,
    rental_reward: float,
):

    request_probs = precompute_poisson_probabilities(max_cars=max_cars[0], rates=request_rates)
    return_probs = precompute_poisson_probabilities(max_cars=max_cars[1], rates=return_rates)

    request_range = (truncated_range(request_probs[0]), truncated_range(request_probs[1]))
    return_range = (truncated_range(return_probs[0]), truncated_range(return_probs[1]))

    # policy = action here
    # The policy can be negative
    returns = 0
    returns -= transfer_cost * np.absolute(policy)

    const_cars_first_loc = min(state[0] - policy, max_cars[0])
    const_cars_second_loc = min(state[1] + policy, max_cars[1])

    # rental requests
    # loop through max cars since at most we can have max cars in the parking lot
    for rental_request_first in request_range[0]:
        for rental_request_second in request_range[1]:

            joint_probability_combination = request_probs[0][rental_request_first] * request_probs[1][rental_request_second]

            cars_first_loc = const_cars_first_loc
            cars_second_loc = const_cars_second_loc

            # Invalidate over limit rental requests
            valid_first = min(cars_first_loc, rental_request_first)
            valid_second = min(cars_second_loc, rental_request_second)

            # Reward for renting
            reward = (valid_first + valid_second) * rental_reward

            cars_first_loc = cars_first_loc - valid_first
            cars_second_loc = cars_second_loc - valid_second

            # Return
            #cars_first_loc = min(cars_first_loc + return_rates[0], max_cars[0])
            #cars_second_loc = min(cars_second_loc + return_rates[1], max_cars[1])

            #returns += joint_probability_combination * (reward + discount_rate * value_function[int(cars_first_loc), int(cars_second_loc)])

            for return_cars_first in return_range[0]: 
                for return_cars_second in return_range[1]:
                    joint_probability_combination_return = return_probs[0][return_cars_first] * return_probs[1][return_cars_second]

                    # Invalidate over limit returns
                    return_cars_first_loc = min(cars_first_loc + return_cars_first, max_cars[0])
                    return_cars_second_loc = min(cars_second_loc + return_cars_second, max_cars[1])

                    joint_probability_combination_whole = joint_probability_combination * joint_probability_combination_return

                    returns += joint_probability_combination_whole * (reward + discount_rate * value_function[int(return_cars_first_loc), int(return_cars_second_loc)])

    return returns



def policy_improvement():
    pass

def policy_iteration(args):

    location_size_1, location_size_2 = args.location_sizes 
    _logger.info(f"Location size 1 is: {location_size_1} and location size 2 is: {location_size_2}")
    evaluation_change_threshold = args.evaluation_change_threshold
    _logger.info(f"Evaluation change threshold: {evaluation_change_threshold}")

    # Value function
    # location size + 1 because we need the number of cars not just the index space
    value_function = np.zeros((location_size_1+1, location_size_2+1))
    _logger.info(f"Initialized value function with shape: {value_function.shape}")
    pprint(value_function)
    # Policies
    # start from the policy that moves no cars and continue
    # a negative number indicates moving cars from second location to first location
    # a positive number indicates moving cars from first location to second location
    #TODO problematic if we have more than two parking lots
    policy = args.initial_policy * np.ones(value_function.shape)
    _logger.info(f"Initialized policies with shape: {policy.shape}")
    pprint(policy)

    transfer_cost = args.transfer_cost
    _logger.info(f"Transfer cost: {transfer_cost}")

    location_return_rates = args.location_return_rates
    _logger.info(f"Location return rates: {location_return_rates}")
    location_request_rates = args.location_request_rates
    _logger.info(f"Location request rates: {location_request_rates}")
    discount_rate = args.discount_rate
    _logger.info(f"Discount rate: {discount_rate}")
    rental_reward = args.rental_reward 
    _logger.info(f"Rental reward: {rental_reward}")

    while True: 

        delta = 0

        for i in range(0, location_size_1 + 1):
            for j in range(0, location_size_2 + 1):

                new_value = policy_evaluation(
                    state=(i,j), 
                    policy=policy[i,j], 
                    transfer_cost=transfer_cost,
                    max_cars=(location_size_1, location_size_2),
                    request_rates=(location_request_rates[0], location_request_rates[1]),
                    return_rates=(location_return_rates[0], location_return_rates[1]),
                    discount_rate=discount_rate,
                    rental_reward=rental_reward,
                    value_function=value_function,
                )

                delta = max(delta, np.absolute(value_function[i,j] - new_value))

                value_function[i,j] = new_value

        if delta < evaluation_change_threshold: 
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy iteration for the Car Rental Problem")

    # Environment parameters
    parser.add_argument(
        "--location-sizes",
        nargs="+",
        default=[20, 20],
        type=int,
        help="Location sizes of the two parking lots. Default value is [20,20].",
    )

    parser.add_argument(
        "--location-return-rates",
        nargs="+",
        default=[3, 2],
        type=int,
        help="Location return rates of the two parking lots. Default value is [3,2].",
    )

    parser.add_argument(
        "--location-request-rates",
        nargs="+",
        default=[3, 4],
        type=int,
        help="Location request rates of the two parking lots. Default value is [3, 4].",
    )

    parser.add_argument(
        "--transfer-cost",
        default=2.0,
        type=float,
        help="Transfer cost between one location and another. Default value is 2.0."
    )

    parser.add_argument(
        "--transfer-limit",
        default=5,
        type=int,
        help="Transfer limit between one location and another. Default value is 5.",
    )

    parser.add_argument(
        "--rental-reward",
        default=10.0,
        type=float,
        help="Reward for renting out a car. Default value is 10.0.",
    )

    # Policy parameters
    parser.add_argument(
        "--discount-rate",
        default=0.9,
        type=float,
        help="Discount rate for future rewards. Default value is 0.9.",
    )

    parser.add_argument(
        "--evaluations-number",
        default=10,
        type=int,
        help="The number of policy evaluations in each policy iteration. Default value is 10.",
    )

    parser.add_argument(
        "--iterations-number",
        default=5,
        type=int,
        help="The number of policy iterations that will be executed. Default value is 5.",
    )

    parser.add_argument(
        "--initial-policy",
        default=0.0,
        type=float,
        help="The initial policy applied to all states. Default value is 0.0.",
    )

    parser.add_argument(
        "--evaluation-change-threshold",
        default=1e-4,
        type=float,
        help="The least amount of change required before stopping the evaluation iteration. Default value is 0.0001.",
    )

    args = parser.parse_args()

    policy_iteration(args)
