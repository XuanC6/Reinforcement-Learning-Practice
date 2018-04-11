# Code Structure
There are 3 classes that define networks.

They are Linear_Q_Network, Deep_Q_Network and Deep_Q_CNN 
that creat linear q network, deep q network and convolutional q network respectively.

There are 2 classes that define DQN_Agent and CNN_Agent respectively.
CNN_Agent is an inheritance of DQN Agent

Agents of "CartPole-v0" and "MountainCar-v0" are created using DQN_Agent
Agent of "SpaceInvaders-v0" are created using CNN_Agent

Functions for training, evaluation and testing, etc are all defined in the AgentClass.
Main function just creat an agent instance, run those functions in order.

Details are covered in the comments of the code.


