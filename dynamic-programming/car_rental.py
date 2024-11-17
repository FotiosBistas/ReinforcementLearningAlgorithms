import argparse
import logging
import numpy as np

from typing import Tuple
from pprint import pprint

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")


def policy_evaluation(
    state: Tuple[int, int],
    policy: int,
):
    pass

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
    policy = np.zeros(value_function.shape)
    _logger.info(f"Initialized policies with shape: {policy.shape}")
    pprint(policy)

    while True: 
        break
        value_change = 0

        for i in range(0, location_size_1):
            for j in range(0, location_size_2):
                policy_evaluation(state=[i,j], policy=policy[i,j])

        if value_change < evaluation_change_threshold: 
            break
    pass

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
