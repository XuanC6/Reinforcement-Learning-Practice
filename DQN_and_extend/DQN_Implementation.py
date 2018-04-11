# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:51:40 2018

@author: CX
"""
import os
import gym
import random
from gym import wrappers
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt


class Linear_Q_Network:
    # This class defines a simple linear Q network with bias
    def __init__(self, network_name, input_state, n_output, dueling = False):
        initializer = tf.random_normal_initializer()
        outputs = tf.layers.dense(input_state, n_output,
                                  kernel_initializer = initializer)
        print("A LQN is created")
        self.Q_values = outputs


class Deep_Q_Network:
    # This class defines Deep Q network with 3 hidden layers
    # dueling is False by default
    def __init__(self, network_name, input_state, n_output, dueling = False):
        initializer = tf.contrib.layers.variance_scaling_initializer()
#        initializer = tf.truncated_normal_initializer()
#        initializer = tf.random_uniform_initializer()
#        initializer = tf.random_normal_initializer()  # choose the initializer here
        activation = tf.nn.elu
#        activation = tf.nn.relu           # choose the activation function here
#        n_hidden_nodes = [6, 6, 4]         # set the number of hidden units here
        n_hidden_nodes = [32, 32, 32]
        
        with tf.variable_scope(network_name) as scope:
            hidden1 = tf.layers.dense(input_state, n_hidden_nodes[0],
                                      activation = activation,
                                      kernel_initializer = initializer)
            hidden2 = tf.layers.dense(hidden1, n_hidden_nodes[1],
                                      activation = activation,
                                      kernel_initializer = initializer)
            hidden3 = tf.layers.dense(hidden2, n_hidden_nodes[2],
                                      activation = activation,
                                      kernel_initializer = initializer)
            if dueling:
                # state value
                S = tf.layers.dense(hidden3, 1,
                                    kernel_initializer = initializer)
                # advantage values
                A = tf.layers.dense(hidden3, n_output,
                                    kernel_initializer = initializer)
                # add them to get the output q values
                # note that let A minus its mean to avoid A just become Q and S = 0
                outputs = S + (A - tf.reduce_mean(A, axis = 1, keepdims = True))
                print("A Dueling DQN is created")
            else:
                outputs = tf.layers.dense(hidden3, n_output,
                                          kernel_initializer = initializer)
                print("A DQN is created")
        self.Q_values = outputs
        # Collect all the variables in this network
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope = scope.name)
        self.params = {param.name[len(scope.name):]: param for param in params}


class DQN_Agent:
    # This class define the agent and methods for training, testing, etc.
    def __init__(self, env_name, QNetwork_Class, max_iter_or_epis,
                 dueling = False):
        # initialize parameters, tensorflow graph
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env2 = gym.make(env_name) # this one is just for evaluation and test
        
        # Set some parameters and hyperparameters
        n_input = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.eps_max = 0.5
        self.eps_min = 0.05
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.QNetwork_Class = QNetwork_Class

        self.eps_iter = 100000 # this means epsilon decay from max to min after this number of iterations
        self.max_iter_or_epis = max_iter_or_epis # maximum training iterations or episodes
        self.eval_updates = 10000 # interval (number of updates) to do a evaluation when training
        self.save_updates = 50000 # interval (number of updates) to save the data in the disk

        self.input_state = tf.placeholder(tf.float32, shape = (None, n_input))
        self.q_network = QNetwork_Class("q_network", 
                                        self.input_state, self.n_actions, dueling)
        
        self.DDQN_on = True   # Whether to use DDQN, default is False
        if self.DDQN_on:
            self.copy_updates = 500 # interval (number of updates) to copy the online network to target
            self.t_network = QNetwork_Class("t_network", 
                                            self.input_state, self.n_actions, dueling)

        # use onehot to select the q value corresponding to the action that is taken
        # thus could compare it to the target value
        self.action_taken = tf.placeholder(tf.int32, shape = (None))
        action_onehot = tf.one_hot(self.action_taken,self.n_actions)
        current_value = tf.reduce_sum(self.q_network.Q_values * action_onehot,
                                      axis = 1)
        self.target_value = tf.placeholder(tf.float32, shape = (None))
        
        # set the loss function
        loss = tf.reduce_mean(tf.square(current_value - self.target_value))
        
        global_step = tf.Variable(0, trainable = False, name = "global_step")
        if env_name == "MountainCar-v0":
            # change and test some parameters for MountainCar
            self.gamma = 1
#            self.learning_rate = 0.001
            self.learning_rate = 0.0001
#            initial_lr = 0.01
#            decay_steps = 100000
#            self.learning_rate = tf.train.polynomial_decay(initial_lr, global_step, 
#                                                           decay_steps, 0.0001)
#            loss = tf.losses.huber_loss(self.target_value, current_value)

         # set the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_step = optimizer.minimize(loss, global_step = global_step)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def sample_batch(self, batch_size = 32):
        # generate index from memory and sample randomly
        batch_index = random.sample(range(len(self.memory)), batch_size)
        batch = [[], [], [], [], []]
        
        # get the batch from the memory using the index above
        for i in batch_index:
            memo = self.memory[i]
            for j in range(len(batch)):
                batch[j].append(memo[j])

        #convert the entries to ndarray for training
        batch = list(map(np.array, batch))
        return batch

    def set_file_path(self,path):
        # set the path to save and restore data of network
        self.path = path
        self.file_path = path + "/saved_data"

    def epsilon_greedy_policy(self, q_values, n_iterations):
        # epsilon greedy policy with decaying epsilon over training
        eps_step = (self.eps_max-self.eps_min) * n_iterations / self.eps_iter
        eps = self.eps_max - eps_step
        eps = max(eps, self.eps_min)

        if np.random.rand() < eps:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(q_values)

    def epsilon_greedy_policy_for_eval(self, q_values):
        # epsilon greedy policy with fixed epsilon for evaluation
        if np.random.rand() < 0.05:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        # greedy policy for test
        return np.argmax(q_values)

    def update_target(self, sess):
        # function to copy the parameter from the online-network to target network
        copy_opt= []
        for param_name, t_param in self.t_network.params.items():
            q_param = self.q_network.params[param_name]
            new_value = sess.run(q_param)
            copy_opt.append(t_param.assign(new_value))
        sess.run(copy_opt)

    def train_replay_memory(self):
        # training function with replay memory
        print("Now train with replay memory")
        self.all_ave_rewards = []
        done = True
        n_updates = 0
        n_iterations = 0
        n_episodes = 0

        # initialize the replay memory using deque from collections
        memory_size = 50000
        burn_in_num = 10000 # number of tuples to burn in memory before training
        self.memory = deque([], maxlen = memory_size)

        burn_in_done = False # is burn_in finished?
        one_third_video_done = False # is this video recorded?
        two_third_video_done = False # is this video recorded?

        with tf.Session() as sess:
            self.init.run()
            # record the initial video
            self.record_one_episode('0_of_3_video')
            if self.DDQN_on:
                self.update_target(sess)

            while True:
                # if the environment is Cartenv_name = "MountainCar-v0"Pole, 
                # then we want to train for certain iterations
                # otherwise we want to train for certain episodes
                criterion = n_episodes
                if self.env_name == "CartPole-v0":
                    criterion = n_iterations
                    
                # break or record the video when certain updates completed
                if criterion >= self.max_iter_or_epis:
                    break
                if criterion == self.max_iter_or_epis//3 and burn_in_done:
                    if not one_third_video_done:
                        self.record_one_episode('1_of_3_video')
                        one_third_video_done = True
                if criterion == (self.max_iter_or_epis//3)*2 and burn_in_done:
                    if not two_third_video_done:
                        self.record_one_episode('2_of_3_video')
                        two_third_video_done = True
                
                if done:
                    if burn_in_done:
                        n_episodes += 1
                    state = self.env.reset()

                current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                        {self.input_state: [state]})
                action = self.epsilon_greedy_policy(current_q_values, n_iterations)

                if burn_in_done:
                    n_iterations += 1
                next_state, reward, done, _ = self.env.step(action)
                
                true_done = done
                if self.env_name == "MountainCar-v0":
                    # For mountaincar, only reach the flag is truly terminal
                    if next_state[0] >= 0.5:
                        true_done = True
                    else:
                        true_done = False

                #append this experience to memory
                self.memory.append((state, action, reward, next_state, true_done))
                state = next_state

                # check if memory has been burned in, if finished, start updating
                if not burn_in_done:
                    if len(self.memory) < burn_in_num:
                        continue
                    else:
                        # if burn_in is done, restart
                        burn_in_done = True
                        done = True
                        continue

                # do memory replay
                    # get the batch from memory
                m_states, m_actions, m_rewards, m_next_states, m_dones = self.sample_batch()
                    # get the next_q_values using next_states in batch
                if self.DDQN_on:
                    next_q_values = self.t_network.Q_values.eval(
                            feed_dict = {self.input_state: m_next_states})
                else:
                    next_q_values = self.q_network.Q_values.eval(
                            feed_dict = {self.input_state: m_next_states})
                    # get the maximum next_q_values for updating
                max_next_q_values = np.max(next_q_values, axis = 1)
                    # get the target value for states and actions
                target_vals = m_rewards+(1-m_dones)*self.gamma*max_next_q_values
                
                # one training update step
                n_updates += 1
                self.train_step.run(feed_dict={
                        self.input_state: m_states, 
                        self.action_taken: m_actions, 
                        self.target_value: target_vals})

                # save the model every few updates
                if n_updates % self.save_updates == 0:
                    self.saver.save(sess, self.file_path)

                if self.DDQN_on:
                    if n_updates % self.copy_updates == 0:
                        self.update_target(sess)
                
                # evaluate the current model every few updates
                if n_updates % self.eval_updates == 0:
                    print(self.env_name)
                    print("current episode:", n_episodes)
                    print("iterations completed:", n_iterations)
                    print("updates completed:", n_updates)
                    self.evaluate()
            
            # save the final model and record the final video
            self.saver.save(sess, self.file_path)
            self.record_one_episode('3_of_3_video')
    
    def train_normal(self):
        # training function without replay memory
        print("Now train without replay memory")
        self.all_ave_rewards = []
        done = True
        n_updates = 0
        n_iterations = 0
        n_episodes = 0
        
        one_third_video_done = False
        two_third_video_done = False
        
        with tf.Session() as sess:

            self.init.run()
            # record the initial video
            self.record_one_episode('0_of_3_video')
            
            while True:
                # if the environment is CartPole, then we want to train for certain iterations
                # otherwise we want to train for certain episodes
                criterion = n_episodes
                if self.env_name == "CartPole-v0":
                    criterion = n_iterations

                # break or record the video when certain updates completed
                if criterion >= self.max_iter_or_epis:
                    break
                if criterion == self.max_iter_or_epis//3 and not one_third_video_done:
                        self.record_one_episode('1_of_3_video')
                        one_third_video_done = True
                if criterion == (self.max_iter_or_epis//3)*2 and not two_third_video_done:
                        self.record_one_episode('2_of_3_video')
                        two_third_video_done = True

                if done:
                    n_episodes += 1
                    state = self.env.reset()

                current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                        {self.input_state: [state]})
                action = self.epsilon_greedy_policy(current_q_values, n_iterations)

                n_iterations += 1
                next_state, reward, done, _ = self.env.step(action)

                # get the next_q_values using next_state
                next_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                             {self.input_state: [next_state]})
                # get the maximum next_q_value for updating
                next_q_value_max = np.max(next_q_values, axis = 1)
                # get the target value for state and aciton
                target_val = reward + (1-done) * self.gamma * next_q_value_max
                if self.env_name == "MountainCar-v0":
                    if next_state[0] >= 0.5:
                        target_val = reward
                    else:
                        target_val = reward + self.gamma * next_q_value_max

                # one training update step
                n_updates += 1
                self.train_step.run(feed_dict = {self.input_state: [state],
                                               self.action_taken: [action], 
                                               self.target_value: target_val})
                state = next_state
                
                # save the model every few updates
                if n_updates % self.save_updates == 0:
                    self.saver.save(sess, self.file_path)
                
                # evaluate the current model every few updates
                if n_updates % self.eval_updates == 0:
                    print(self.env_name)
                    print("current episode:", n_episodes)
                    print("iterations completed:", n_iterations)
                    print("updates completed:", n_updates)
                    self.evaluate()
            
            # save the final model and record the final video
            self.saver.save(sess, self.file_path)
            self.record_one_episode('3_of_3_video')

    def train(self, replay_memory_on = False):
        # choose function for training
        if replay_memory_on:
            self.train_replay_memory()
        else:
            self.train_normal()

    def evaluate(self):
        # function to evaluate the model when training
        # record the average reward over 20 episodes using epsilon (0.05) greedy
        n_episodes = 0
        done = True
        total_reward = 0
        all_rewards = []

        while True:
            if done:
                n_episodes += 1
                if n_episodes > 1:
                    all_rewards.append(total_reward)
                state = self.env2.reset()
                total_reward = 0
                
            if n_episodes > 20:
                average_reward = sum(all_rewards)/20
                self.all_ave_rewards.append(average_reward)
                print("average reward over 20 episodes:",average_reward)
                print('')
                break
            
            current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                    {self.input_state: [state]})
            action = self.epsilon_greedy_policy_for_eval(current_q_values)
            next_state, reward, done, _ = self.env2.step(action)
            total_reward += reward
            state = next_state

    def test(self):
        # function to test the final model
        # record all rewards from 100 episodes using greedy policy
        if not os.path.isfile(self.file_path + '.index'):
            print("No existing weights to test!")
            return
        
        n_episodes = 0
        done = True
        total_reward = 0
        self.all_test_rewards = []
        
        with tf.Session() as sess:
            
            self.saver.restore(sess, self.file_path)
            
            while True:
                if done:
                    n_episodes += 1
                    if n_episodes > 1:
                        self.all_test_rewards.append(total_reward)
                    state = self.env2.reset()
                    total_reward = 0
                    
                if n_episodes > 100:
                    break
                
                current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                        {self.input_state: [state]})
                action = self.greedy_policy(current_q_values)
                next_state, reward, done, _ = self.env2.step(action)
                total_reward += reward
                state = next_state

    def record_one_episode(self, video_path):
        # function to record the video of one episode using greedy policy
        path = self.path + '/' + video_path + '_' + self.env_name
        env = wrappers.Monitor(self.env2, path, force=True)
        state = env.reset()
        while True:
            current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                    {self.input_state: [state]})
            action = self.greedy_policy(current_q_values)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break

    def plot_evaluations(self):
        # function to save the plot of evaluation results
        plt.figure()
        plt.plot(self.all_ave_rewards)
        plt.title("Average Rewards of Evaluation of " + self.env_name)
        plt.savefig(self.path + "/Average_Rewards_of_Evaluation.png")
        plt.clf()
        
    def plot_test(self):
        # function to save the plot of test results
        plt.figure()
        plt.plot(self.all_test_rewards)
        plt.title("Test Rewards for 100 episodes of " + self.env_name)
        plt.savefig(self.path + "/Test_Rewards_for_100_episodes.png")
        plt.clf()

'''
Main Function Part

'''
def main(env_name, file_path, QNetwork_Class, replay_memory_on,
         max_iter_or_epis, dueling, do_train = True):
    
    tf.reset_default_graph()
    # creat the agent
    my_agent = DQN_Agent(env_name, QNetwork_Class, max_iter_or_epis, dueling) 
    
    # set the file path
    my_agent.set_file_path(file_path)

    # do training and evaluation
    if do_train:
        my_agent.train(replay_memory_on)
        my_agent.plot_evaluations()
    
    # do testing
    my_agent.test()
    my_agent.plot_test()

    # analyze the testing result
    test_rewards = np.array(my_agent.all_test_rewards)
    test_mean = np.mean(test_rewards)
    test_std = np.std(test_rewards)

    print("final reward mean: {}; std: {}".format(test_mean, test_std))


if __name__ == '__main__':
                                # choose the environment
    env_name = "CartPole-v0"
#    env_name = "MountainCar-v0"
    file_path = "./" + env_name  # path to save the model, videos and plots

#    QNetwork_Class = Linear_Q_Network # choose the network (LQN or DQN)
    QNetwork_Class = Deep_Q_Network
    
    replay_memory_on = 1 # whether to use replay memory
    dueling = 0          # whether to use dueling architecture

    max_iter_or_epis = 200000  # maximum training iterations or episodes
    do_train = 1         # whether to train the model

    main(env_name, file_path, QNetwork_Class, replay_memory_on,
         max_iter_or_epis, dueling, do_train)

