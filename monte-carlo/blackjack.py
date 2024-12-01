import argparse
import logging
import numpy as np

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

def estimate(args):

    epsilon = args.epsilon
    episodes = args.episodes_per_policy_update

    # 1.) The player always hits before and after 12. The first element represents the sum - 12 [0 -> 12 ... 9 -> 21]
    # 2.) The dealer shows only their first card 1-10
    # 3.) The player has either a usable ace or not [0|1]
    # 4.) The action the player will take hit or stick [0|1]
    initial_state_action_values = np.zeros((10,10,2,2))

    # Player sticks on 20,21 hits on the rest
    initial_policy_player = np.ones(22, dtype=int)
    initial_policy_player[20] = STICK
    initial_policy_player[21] = STICK
    # Dealer stick after 17 sum
    policy_dealer = np.ones(22, dtype=int)
    policy_dealer[17:] = STICK

    for episode in range(episodes): 
        _logger.debug(f"Running episode {episode}")

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