import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt


from pprint import pprint

logging.basicConfig(
    format='%(filename)s - %(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)


_logger = logging.getLogger()
_logger.setLevel("DEBUG")

def plot_grid_values(x, name="grid_values.png"):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(x, cmap="coolwarm", interpolation="nearest")

    # Loop over data dimensions and create text annotations.
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ax.text(j, i, f"{x[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8, fontweight='bold')

    # Add color bar
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

    # Labels and formatting
    ax.set_title("Grid Values of x (Result Array)")
    ax.set_xticks(np.arange(x.shape[1]))
    ax.set_yticks(np.arange(x.shape[0]))
    ax.set_xticklabels(range(x.shape[1]))
    ax.set_yticklabels(range(x.shape[0]))

    plt.show()
    plt.savefig(f"./{name}")


def plot_a_matrix(matrix):
    fig, ax = plt.subplots(figsize=(16, 16))  # Significantly larger figure size
    cax = ax.matshow(matrix, cmap="coolwarm", aspect='auto')  # Adjust aspect to 'auto' for more space

    fig.colorbar(cax)

    # Annotate each cell with the value if it is non-zero
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value != 0:  # Skip zero values to reduce clutter
                ax.text(j, i, f"{value:.3f}", va="center", ha="center", color="black", fontsize=6)  # Small font size

    ax.set_title("Heatmap of A Matrix", fontsize=16)  # Larger title font
    plt.xlabel("Columns", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig("./a_matrix.png")


# Plot for b_matrix
def plot_b_matrix(vector):
    fig, ax = plt.subplots(figsize=(10, 2))
    cax = ax.matshow(vector.reshape(1, -1), cmap="coolwarm", aspect="auto")
    fig.colorbar(cax, orientation="horizontal")

    # Annotate each cell with the value
    for i in range(len(vector)):
        ax.text(i, 0, f"{vector[i]:.2f}", va="center", ha="center", color="black", fontsize=8)

    ax.set_title("Heatmap of B Matrix (Vector)")
    plt.xlabel("Index")
    plt.yticks([])  # Hide y-axis ticks for single row
    plt.show()
    plt.savefig("./b_matrix.png")

def run(arguments):

    #Horizontal dimension
    n = args.n
    _logger.info(f"N (Horizontal Dimension): {n}")
    #Vertical dimension
    m = args.m 
    _logger.info(f"M (Vertical Dimension): {m}")
    #Default reward for in bound state transition
    stay_reward = args.stay_reward
    _logger.info(f"Stay reward: {stay_reward}")
    #Default reward in out bound state transition
    drop_reward = args.drop_reward
    _logger.info(f"Drop reward: {drop_reward}")
    #Discount factor 
    gamma = args.gamma
    _logger.info(f"Gamma: {gamma}")
    #Convert the input to a list of tuples (s_x_coordinate, s_y_coordinate, reward)
    #Parse special rewards into a dictionary
    if len(arguments.special_rewards) != 0:
        special_rewards = {
            (int(x), int(y)): int(r) for x, y, r in 
            (reward.split(",") for reward in arguments.special_rewards)
        }
    else:
        special_rewards = {}

    if any([True for x, y in special_rewards.keys() if (x >= n or y >= m) or (x < 0 or y < 0)]): 
        _logger.error("Invalid special reward indexes given")
        return 

    _logger.info(f"Special rewards: {special_rewards}")

    random_walk_distribution = args.random_walk_distribution
    _logger.info(f"Random walk distribution: {random_walk_distribution}")

    teleports = args.teleports
    if len(arguments.teleports) != 0:
        teleports = {
            (int(x), int(y)): (int(z), int(w)) for x, y, z, w  in 
            (coordinates.split(",") for coordinates in arguments.teleports)
        }
    else:
        teleports = {}

    _logger.info(f"Teleports: {teleports}")

    #define the actions left, right, up, down
    #this will be added to a position inside an array
    actions = [
        np.array([0, -1]),
        np.array([0, 1]),
        np.array([-1, 0]),
        np.array([1, 0]),
    ]

    #number of states 
    number_of_states = n * m
    #Solving a system of linear equations for each state
    I = np.eye(number_of_states)
    #Transition probabilities 
    a_matrix = np.zeros((number_of_states, number_of_states))
    #Rewards
    b_matrix = np.zeros(number_of_states)

    #solve the linear equations
    for i in range(n):
        for j in range(m):  
            flat_state_index = np.ravel_multi_index([i, j], (n, m))
            for action_index, action in enumerate(actions, start=0):

                #get current state                
                current_state = (i,j)
                #perform the step
                s_prime = np.array([i,j]) + action
                #reward for going to a neutral state
                reward = stay_reward

                if current_state in teleports: 

                    if (current_state[0], current_state[1]) in special_rewards:
                        #override the reward with the special one
                        reward = special_rewards[(current_state[0], current_state[1])]

                    teleport_destination = teleports[(current_state[0], current_state[1])]
                    #override s_prime with the teleport
                    s_prime = np.array(teleport_destination)


                # if x, y are outside the gridworld's bound set the reward
                if (s_prime[0] >= n or s_prime[0] < 0) or (s_prime[1] >= m or s_prime[1] < 0):
                    reward = drop_reward
                    # state should remain the same if we are out of bounds
                    s_prime = np.array([i,j])


                flat_s_prime_index = np.ravel_multi_index(s_prime, (n, m))
                a_matrix[flat_state_index, flat_s_prime_index] += random_walk_distribution[action_index] * gamma
                b_matrix[flat_state_index] += random_walk_distribution[action_index] * reward

    temp = I - a_matrix

    x = np.linalg.solve(temp, b_matrix)
    x = x.reshape(n, m)
    #x = np.round(x, decimals=1)

    _logger.info("Result A matrix:")
    pprint(temp)
    _logger.info("Result b matrix:")
    pprint(b_matrix)
    _logger.info("Result matrix:")
    pprint(x)

    plot_grid_values(x=x)
    plot_a_matrix(matrix=temp)
    plot_b_matrix(vector=b_matrix)



if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Calculate a nxm gridworld rewards")

    parser.add_argument(
        "--n", 
        type=int, 
        default=5, 
        help="Horizontal dimension of the Gridworld. Default value is 5."
    )

    parser.add_argument(
        "--m", 
        type=int,  
        default=5, 
        help="Vectical dimension of the Gridworld. Default value is 5."
    )

    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.9, 
        help="The gamma value of the value function. Default value is 0.9."
    )

    parser.add_argument(
        "--random-walk-distribution", 
        type=float, 
        nargs="+", 
        default=[0.25, 0.25, 0.25, 0.25], 
        help="The distribution of the random walk actions. Default value is uniform."
    )

    parser.add_argument(
        "--stay-reward", 
        type=float, 
        default=0, 
        help="The default reward for a pair(s,a) that does not lead the agent outside the gridworld. Default value is 0."
    )

    parser.add_argument(
        "--drop-reward", 
        type=float, 
        default=-1, 
        help="The default reward for a pair(s,a) that does lead the agent outside the gridworld. Default value is -1."
    )

    parser.add_argument(
        "--special-rewards",
        type=str,
        nargs="+",
        default=["0,1,10","0,3,5"], 
        help="List of special rewards in the format x,y,reward for each state. Default value is ['0,1,10','0,3,5']", 
    )

    parser.add_argument(
        "--teleports",
        type=str,
        nargs="+",
        default=["0,1,4,1", "0,3,2,3"],
        help="List of teleports in the format x,y,z,w for each state. Default value is ['0,1,4,1', '0,3,2,3']",
    )

    args = parser.parse_args()

    run(arguments=args)

