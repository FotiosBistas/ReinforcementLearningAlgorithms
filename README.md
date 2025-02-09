# ReinforcementLearningAlgorithms

This repository contains implementations of key algorithms in reinforcement learning, following the approaches outlined in *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto. It is structured to explore and experiment with various RL techniques, including dynamic programming, Monte Carlo methods, temporal-difference learning, and more.

## Repository Structure

The repository is organized into folders, each corresponding to a specific set of reinforcement learning algorithms or concepts:

### 1. **dynamic-programming**
   - Contains implementations of algorithms such as policy iteration and value iteration.
   - These methods rely on a complete knowledge of the environment's dynamics (state transition probabilities and rewards).
### 2. **k-armed-bandits**
   - Explores the k-armed bandit problem.
   - Includes ε-greedy methods, optimistic initialization, and UCB (Upper Confidence Bound).
### 3. **monte-carlo**
   - Focuses on Monte Carlo methods for estimating value functions based on sample episodes.
   - Includes examples of first-visit Monte Carlo methods.
### 4. **off-policy-monte-carlo**
   - Implements off-policy Monte Carlo methods.
### 5. **temporal-difference-methods**
   - Features TD(0), SARSA, and Q-learning algorithms.

etc. 

## Prerequisites

To run the code in this repository, you need the following dependencies:

- Python 3.x
- Libraries: `numpy`, `matplotlib`, and optionally `pandas` for data manipulation

You can install the dependencies using:
```bash
pip install -r requirements.txt
