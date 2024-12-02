import argparse
import logging
import numpy as np
import numpy.typing as npt 

from pprint import pprint
from typing import Tuple

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")

HIT = 1
STICK = 0

def exploring_starts(): 
    pass

def draw_card():
    card = min(np.random.choice(range(1, 14), 1), 10)
    return card

def argmax(
    state: Tuple[int, int, int],
    action_values: npt.NDArray[np.float64], 
    action_values_counts: npt.NDArray[np.float64],
):

    # decouple the state
    # State is whether the player has an ACE
    # A-10 the card of the dealer
    # 12...21 the current hand
    usable_ace = state[0]
    player_card = state[1]
    dealer_card = state[2]

    temp = action_values[player_card, dealer_card, usable_ace, :] / action_values_counts[player_card, dealer_card, usable_ace, :]

    # Find all indices of the maximum value
    indices = np.where(temp == np.max(temp))[0]

    # Randomly select one of them
    return np.random.choice(indices)


def play_round():
    pass

def estimate(args):

    epsilon = args.epsilon
    episodes = args.episodes_per_policy_update

    # 1.) The player always hits before and after 12. The first element represents the sum - 12 [0 -> 12 ... 9 -> 21]
    # 2.) The dealer shows only their first card 1-10
    # 3.) The player has either a usable ace or not [0|1]
    # 4.) The action the player will take hit or stick [0|1]
    initial_state_action_values = np.zeros((10,10,2,2))
    _logger.info("Initial state action values")
    print(initial_state_action_values)
    # Keep in track how many times a state has been visited in order to
    # calculate the average. Initialize to one due to division
    initial_state_action_values_counts = np.ones((10,10,2,2))

    # Player sticks on 20,21 hits on the rest
    policy_player = np.ones(22, dtype=int)
    policy_player[20] = STICK
    policy_player[21] = STICK
    _logger.info("Policy player")
    pprint(policy_player)
    # Dealer stick after 17 sum
    policy_dealer = np.ones(22, dtype=int)
    policy_dealer[17:] = STICK
    _logger.info("Policy dealer")
    pprint(policy_dealer)

    for episode in range(episodes): 
        # Initialize a random state
        # State is whether the player has an ACE
        # 12...21 the current hand
        # A-10 the card of the dealer
        initial_state = [
            np.random.choice(range(0,1)), 
            np.random.choice(range(12, 22)), 
            np.random.choice(range(1,11)),
        ]

        initial_action = np.random.choice([HIT, STICK])
        #_logger.debug(f"Running episode {episode}")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Calculate an optimal policy based using Monte Carlo")

    parser.add_argument(
        "--episodes-per-policy-update",
        type=int,
        default=10000,
        help="Number of simulated episodes per policy update. Default value is 10_000.",
    )

    #parser.add_argument(
    #    "--total-policy-updates",
    #    type=int,
    #    default=,
    #)

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
    )
    
    args = parser.parse_args()

    estimate(args)