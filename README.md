# Multi Armed Bandit Study
A study in the comparison of multi armed bandit strategies including a proof of concept for a machine learning approach

## Strategy Legend:

This repository references four distinct strategies for the multi armed bandit problem. 

1. Random play (plotted in Red)
 - As the name implies this strategy entails choosing random bandits,
 - It is plotted along with the other strategies for comparative purposes
 
 2. Perfect play (plotted in blue)
- This entails playing the bandit with the highest expected value every time
- Given that this is hidden knowledge in the problem this strategy is also plotted for comparative purposes

3. Upper Confidence Bound Strategy (plotted in yellow)
- This entails calculating the upper confidence bound for each bandit

4. Neural Network Greedy with Neuro-Evolution (plotted in Green)
- This entails evolving a neural network to compute confidence bounds based on available information
- This is approach is the focus of this repository.

## Neuro Evolution Methodoloy:

- Intuition: 

The intuition for this strategy is that the optimal greedy algorithm for the multi armed bandit problem (if such a thing is presumed to exist) would be a function of all the available information during the simulation. The available information in this problem consists of **the time step/number of plays**, **the number of plays per bandit** and **the reward per bandit**. We can therfore use the as inputs to a neural network where the output is a score that will be used as our greedy choice. The simplest method for training a neural network to do this succesfully is to engage in genetic neuro-evolution across a number of simulations. 

- Methods:

During the simulation stage a random parent neural network is instantiated. A set of children are evolved from this parent with a mutation rate (initialized to 35% or 0.35) that is used to linearly augment the weights and biases of the children in a randomized manner. 

This generation is tested in a new multi armed bandit simulation and the best child becomes the parent for the next generation. After all generations have been tested the final parent is saved and returned. 



## Results:

The strategies were tested in a small validation simulation with 500 timesteps (on the x-axis) against cummulative reward (on the y-axis), shown below...


<img width="554" alt="Screen Shot 2023-04-29 at 5 20 06 PM" src="https://user-images.githubusercontent.com/108235294/235324723-7981caf1-8136-4ae4-8855-7b00703bf3ad.png">


As can be seen above the Neural Network strategy approximated the UCB strategy very closely essentially implying that the neural network found that the optimal solution given its inputs was to approximate the UCB algorithm.

## Details:
- Network architecture: 4 layers of shape [3,5,5,5,5,1]
- Randomization rate: 0.35
- Generation size: 10
- Number of generations: 100
- Tests per generation: 3
- Training Bandits: 100
- Training Plays/tickets: 1000
- Validation Bandits: 50
- Validation Play/tickets: 500
- Mutation Decay Rate: 0.99
