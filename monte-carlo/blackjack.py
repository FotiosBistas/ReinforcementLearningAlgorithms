import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.typing as npt 

from pprint import pprint
from typing import Tuple, Union, Callable
from tqdm import tqdm

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")

HIT = 1
STICK = 0

# Player sticks on 20,21 hits on the rest
policy_player = np.ones(22, dtype=int)
policy_player[20] = STICK
policy_player[21] = STICK
_logger.info("Policy player")
#pprint(policy_player)

# Dealer stick after 17 sum
policy_dealer = np.ones(22, dtype=int)
policy_dealer[17:] = STICK
_logger.info("Policy dealer")
#pprint(policy_dealer)

def exploring_starts(): 
    pass

def draw_card():
    card = min(np.random.choice(range(1, 14), 1)[0], 10)
    return card

def argmax(
    state: Tuple[int, int, int],
    action_values: npt.NDArray[np.float64], 
    action_values_counts: npt.NDArray[np.float64],
) -> int:

    # decouple the state
    # State is whether the player has an ACE
    # A-10 the card of the dealer
    # 12...21 the current hand
    usable_ace = state[0]
    player_card_index = state[1] - 12
    dealer_card_index = state[2] - 1

    temp = action_values[player_card_index, dealer_card_index, usable_ace, :] / action_values_counts[player_card_index, dealer_card_index, usable_ace, :]

    # Find all indices of the maximum value
    indices = np.where(temp == np.max(temp))[0]

    # Randomly select one of them
    return np.random.choice(indices)

def target_policy(sum: int, player_not_dealer: bool) -> int:

    if(player_not_dealer): 
        return policy_player[sum]

    return policy_dealer[sum]


def actual_card_value(card: int) -> int:
    return 11 if card == 1 else card


def play_round(
    policy: Callable,
    initial_state: Tuple[int, int, int], 
    initial_action: int,
    state_action_values: npt.NDArray[np.float64],
    state_action_values_counts: npt.NDArray[np.int64],
) -> Tuple[int, int, list]:

    state_sequence = []

    # Draw dealer card in order to check for an ACE + 10
    # Reminder the state 
    # State is whether the player has an ACE
    # 12...21 the current hand
    # A-10 the card of the dealer
    first_dealer_card = initial_state[2]
    second_dealer_card = draw_card()

    dealer_sum = actual_card_value(first_dealer_card) + actual_card_value(second_dealer_card)
    dealer_usable_ace = 1 == first_dealer_card or 1 == second_dealer_card

    # holds two aces the sums is larger than 21
    if dealer_sum == 22: 
        dealer_sum -= 10

    player_usable_ace = initial_state[0]
    player_sum = initial_state[1]
    action = initial_action

    player_ace_count = 1 if player_usable_ace else 0 

    # Player turn
    # If not go bust count the ace as 11
    while True: 
        
        state_sequence.append([(player_usable_ace, player_sum, first_dealer_card), action])

        if action == STICK: 
            break

        if policy.__name__ == "argmax": 
            action = policy(
                state=(player_usable_ace, player_sum, first_dealer_card),
                action_values=state_action_values,
                action_values_counts=state_action_values_counts,
            )
        elif policy.__name__ == "target_policy":
            action = policy(
                sum=player_sum, 
                player_not_dealer=True,
            )

        new_card = draw_card()

        if new_card == 1: 
            player_ace_count += 1

        # The player should try to always use an ACE as an 11 first
        player_sum += actual_card_value(new_card)

        # Use 1 instead of ACE if player busts using the ace as 11
        while player_sum > 21 and player_ace_count: 
            player_sum -= 10
            player_ace_count -= 1
        

        if player_sum > 21:
            return initial_state, -1, state_sequence

        player_usable_ace = int(player_ace_count >= 1 and player_sum + 10 <= 21)

    dealer_ace_count = 1 if dealer_usable_ace else 0
    # Dealer Turn
    while True:
        action = policy_dealer[dealer_sum]

        if action == STICK:
            break

        new_card = draw_card()

        if new_card == 1:
            dealer_ace_count += 1

        # The player should try to always use an ACE as an 11 first
        dealer_sum += actual_card_value(new_card)

        # Use 1 instead of ACE if player busts using the ace as 11
        while dealer_sum > 21 and dealer_ace_count:
            dealer_sum -= 10
            dealer_ace_count -= 1

        # dealer busts
        if dealer_sum > 21:
            return initial_state, 1, state_sequence

        dealer_usable_ace = int(dealer_ace_count >= 1 and dealer_sum + 10 <= 21)

    if player_sum > dealer_sum:
        return initial_state, 1, state_sequence
    elif player_sum == dealer_sum:
        return initial_state, 0, state_sequence
    else:   
        return initial_state, -1, state_sequence


def estimate(args):

    episodes = args.episodes_per_policy_update

    # 1.) The player always hits before and after 12. The first element represents the sum - 12 [0 -> 12 ... 9 -> 21]
    # 2.) The dealer shows only their first card 1-10
    # 3.) The player has either a usable ace or not [0|1]
    # 4.) The action the player will take hit or stick [0|1]
    state_action_values = np.zeros((10,10,2,2))
    _logger.info("Initial state action values")
    # Keep in track how many times a state has been visited in order to
    # calculate the average. Initialize to one due to division
    state_action_values_counts = np.ones((10,10,2,2))


    for episode in tqdm(range(episodes)): 
        # Initialize a random state
        # State is whether the player has an ACE
        # 12...21 the current hand
        # A-10 the card of the dealer
        initial_state = [
            np.random.choice(range(0,2)), 
            np.random.choice(range(12, 22)), 
            np.random.choice(range(1,11)),
        ]

        initial_action = np.random.choice([HIT, STICK])
        #_logger.debug(f"Running episode {episode}")

        current_policy = argmax

        # If first episode initialize with the target policy 
        if episode == 0: 
            current_policy = target_policy

        s, reward, sequence = play_round(
            policy=current_policy,
            initial_action=initial_action, 
            initial_state=initial_state,
            state_action_values=state_action_values,
            state_action_values_counts=state_action_values_counts,
        )

        first_visit = set()

        for (player_usable_ace, player_sum, dealer_card), action in sequence:
            player_card_index = player_sum - 12
            dealer_card_index = dealer_card - 1
            state_action_pair = (player_usable_ace, player_card_index, dealer_card_index, action)
            # first visit 
            # don't visit the same states inside the same episode
            if state_action_pair in first_visit:
                continue

            first_visit.add(state_action_pair)

            state_action_values[player_card_index, dealer_card_index, player_usable_ace, action] += reward
            state_action_values_counts[player_card_index, dealer_card_index, player_usable_ace, action] += 1

    return state_action_values, state_action_values_counts

def create_5_1(
    states_usable_ace, 
    states_no_usable_ace, 
    episodes,
):

    states = [states_usable_ace, states_no_usable_ace]

    titles = [f'Usable Ace, {episodes} Episodes', f'No Usable Ace, {episodes} Episodes']

    _, axes = plt.subplots(2, 1, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(np.flipud(state), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig(f'./images/{titles}.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blackjack: Monte Carlo Exploring Starts")
    parser.add_argument("--episodes-per-policy-update", type=int, default=10000, help="Number of episodes")
    args = parser.parse_args()

    # Run Monte Carlo Exploring Starts
    _logger.info("Running Monte Carlo Exploring Starts...")

    state_action_values, state_action_values_counts = estimate(args)

    no_usable = np.max(state_action_values[:, :, 0, :] / state_action_values_counts[:, :, 0, :], axis=-1)
    usable = np.max(state_action_values[:, :, 1, :] / state_action_values_counts[:, :, 1, :], axis=-1)

    create_5_1(
        states_no_usable_ace=no_usable, 
        states_usable_ace=usable, 
        episodes=args.episodes_per_policy_update
    )

    # Compute optimal policy and state-value function
    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)
    # get the optimal policy
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)




