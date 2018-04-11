#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 15:24:36 2018

@author: xuan
"""
import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.transform import resize
from DQN_Implementation import DQN_Agent

class Deep_Q_CNN:
    # This class defines CNN Q network for the atari game
    # follows the structure in the paper "Playing Atari..."
    def __init__(self, network_name, input_state, n_output, dueling = False):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        activation = tf.nn.relu
        with tf.variable_scope(network_name) as scope:
            hidden1 = tf.layers.conv2d(inputs = input_state, filters = 16, 
                                       kernel_size = (8,8), strides = 4, 
                                       padding = 'SAME', activation = activation, 
                                       kernel_initializer = initializer)
            hidden2 = tf.layers.conv2d(inputs = hidden1, filters = 32, 
                                       kernel_size = (4,4), strides = 2, 
                                       padding = 'SAME', activation = activation, 
                                       kernel_initializer = initializer)
            flat = tf.reshape(hidden2, shape = (-1, 3872))
            hidden3 = tf.layers.dense(flat, 256, activation = activation, 
                                       kernel_initializer = initializer)
            outputs = tf.layers.dense(hidden3, n_output,
                                      kernel_initializer = initializer)
        
        print("A CNN is created")
        self.Q_values = outputs
        # Collect all the variables in this network
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope = scope.name)
        self.params = {param.name[len(scope.name):]: param for param in params}


class CNN_Agent(DQN_Agent):
    # This class define the agent and methods for training, testing, etc.
    def __init__(self, env_name, QNetwork_Class, max_iter_or_epis, dueling = False):
        # initialize parameters, tensorflow graph
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env2 = gym.make(env_name)

        self.n_actions = self.env.action_space.n
        self.eps_max = 1
        self.eps_min = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.0001

        self.eps_iter = 1000000 # this means epsilon decay from max to min after this number of iterations
        self.max_iter_or_epis = max_iter_or_epis # maximum training episodes
        self.eval_updates = 20000 # interval (number of updates) to do a evaluation when training
        self.copy_updates = 10000 # interval (number of updates) to copy the online network to target network
        
        self.input_state = tf.placeholder(tf.float32, shape = (None, 84, 84, 4))

        self.q_network = Deep_Q_CNN("q_network", 
                                    self.input_state, self.n_actions)
        self.t_network = Deep_Q_CNN("t_network", 
                                    self.input_state, self.n_actions)
        # use onehot to select the q value corresponding to the action that is taken
        # thus could compare it to the target value
        self.action_taken = tf.placeholder(tf.int32, shape = (None))
        action_onehot = tf.one_hot(self.action_taken,self.n_actions)
        current_value = tf.reduce_sum(self.q_network.Q_values * action_onehot,
                                      axis = 1)
        self.target_value = tf.placeholder(tf.float32, shape = (None))
        
        # set the loss and optimizer
        loss = tf.reduce_mean(tf.square(current_value - self.target_value))
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        
        self.train_step = optimizer.minimize(loss)
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

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
        print("Now train with replay memory CNN")
        self.all_ave_rewards = []
        done = True
        n_updates = 0
        n_iterations = 0
        n_episodes = 0
        frames = []
        
        # initialize the replay memory using deque from collections
        memory_size = 1000000
        self.memory = deque([],maxlen = memory_size)
        burn_in_num = 10000 # number of tuples to burn in memory before training
        burn_in_done = False # is burning in finished

        one_third_video_done = False # is this video recorded?
        two_third_video_done = False # is this video recorded?

        with tf.Session() as sess:
            self.init.run()
            # record the initial video
            self.record_one_episode('0_of_3_video')
            self.update_target(sess)

            while True:
                # break or record the video when certain updates completed
                if n_iterations >= self.max_iter_or_epis:
                    break
                if n_iterations == self.max_iter_or_epis//3 and burn_in_done:
                    if not one_third_video_done:
                        self.record_one_episode('1_of_3_video')
                        one_third_video_done = True
                if n_iterations == (self.max_iter_or_epis//3)*2 and burn_in_done:
                    if not two_third_video_done:
                        self.record_one_episode('2_of_3_video')
                        two_third_video_done = True

                if done:
                    done = False
                    if burn_in_done:
                        n_episodes += 1
                    img = self.env.reset()
                    frames = []

                if len(frames) < 4:
                    # keep add frame until 4
                    frames.append(CNN_Agent.preprocess_state(img))
                    continue

                # transpose from 4x84x84 to 84x84x4
                np_frames = np.array(frames).transpose([1,2,0]) 
                current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                        {self.input_state: [np_frames]})
                action = self.epsilon_greedy_policy(current_q_values, n_iterations)
                
                if burn_in_done:
                    n_iterations += 1
                next_img, reward, done, _ = self.env.step(action)
                
                next_frames = frames
                next_frames.append(CNN_Agent.preprocess_state(next_img))
                next_frames.pop(0)
                np_next_frames = np.array(next_frames).transpose([1,2,0])
                #append this experience to memory
                self.memory.append((np_frames, action, reward, np_next_frames, done))
                frames = next_frames

                # check if memory has been burned in, if finished, start updating 
                # otherwise we want to train for certain episodes
                if not burn_in_done:
                    if len(self.memory) < burn_in_num:
                        continue
                    else:
                        burn_in_done = True
                        done = True
                        continue

                # do memory replay
                    # get the batch from memory
                m_frames, m_actions, m_rewards, m_next_frames, m_dones = self.sample_batch()
                    # get the next_q_values using next_states in batch
                next_q_values = self.q_network.Q_values.eval(
                        feed_dict = {self.input_state: m_next_frames})
                    # get the maximum next_q_values for updating
                max_next_q_values = np.max(next_q_values, axis = 1)
                    # get the target value for states and actions
                target_vals = m_rewards+(1-m_dones)*self.gamma*max_next_q_values
                
                # one training update step
                n_updates += 1
                self.train_step.run(feed_dict={
                        self.input_state: m_frames, 
                        self.action_taken: m_actions, 
                        self.target_value: target_vals})
                    
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
            self.test()
    
    def evaluate(self):
        # function to evaluate the model when training
        # record the average reward over 20 episodes using epsilon (0.05) greedy
        n_episodes = 0
        done = True
        total_reward = 0
        all_rewards = []
        frames = []

        while True:
            if done:
                done = False
                n_episodes += 1
                if n_episodes > 1:
                    all_rewards.append(total_reward)
                img = self.env2.reset()
                frames = []
                total_reward = 0
                
            if n_episodes > 50:
                average_reward = sum(all_rewards)/20
                self.all_ave_rewards.append(average_reward)
                print("average reward over 20 episodes:",average_reward)
                print('')
                break
            
            if len(frames) < 4:
                frames.append(CNN_Agent.preprocess_state(img))
                continue

            np_frames = np.array(frames).transpose([1,2,0])
            current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                    {self.input_state: [np_frames]})
            action = self.epsilon_greedy_policy_for_eval(current_q_values)
            next_img, reward, done, _ = self.env2.step(action)
            next_frames = frames
            next_frames.append(CNN_Agent.preprocess_state(next_img))
            next_frames.pop(0)
            total_reward += reward
            frames = next_frames
    
    def test(self):
        # function to test the final model
        # record all rewards from 100 episodes using greedy policy
        n_episodes = 0
        done = True
        total_reward = 0
        self.all_test_rewards = []
        frames = []

        while True:
            if done:
                done = False
                n_episodes += 1
                if n_episodes > 1:
                    self.all_test_rewards.append(total_reward)
                img = self.env2.reset()
                frames = []
                total_reward = 0
                
            if n_episodes > 100:
                break
            
            if len(frames) < 4:
                frames.append(CNN_Agent.preprocess_state(img))
                continue

            np_frames = np.array(frames).transpose([1,2,0])
            current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                    {self.input_state: [np_frames]})
            action = self.greedy_policy(current_q_values)
            next_img, reward, done, _ = self.env2.step(action)
            next_frames = frames
            next_frames.append(CNN_Agent.preprocess_state(next_img))
            next_frames.pop(0)
            total_reward += reward
            frames = next_frames
    
    def record_one_episode(self, video_path):
        # function to record the video of one episode using greedy policy
        path = self.path + '/' + video_path + '_' + self.env_name
        env = wrappers.Monitor(self.env2, path, force=True)
        img = env.reset()
        frames = []
        while True:
            if len(frames) < 4:
                frames.append(CNN_Agent.preprocess_state(img))
                continue

            np_frames = np.array(frames).transpose([1,2,0])
            current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                    {self.input_state: [np_frames]})
            action = self.greedy_policy(current_q_values)
            next_img, _, done, _ = env.step(action)
            next_frames = frames
            next_frames.append(CNN_Agent.preprocess_state(next_img))
            next_frames.pop(0)
            frames = next_frames
            if done:
                break

    @classmethod
    def preprocess_state(cls, img):
        # function to process the image returned by the env
        img = img[25:200]           # cut the useless top and bottom of the image
        img = resize(img, (84,84,3), mode = 'reflect') # resize the image to 84x84
        img = img.mean(axis = 2)    # get the gray image
        img = img / 256             # normalize to 0-1
        return img


'''
Main Function Part

'''
def main(env_name, file_path, QNetwork_Class, replay_memory_on,
         max_iter_or_epis, dueling, do_train = True):
    
    tf.reset_default_graph()
    # creat the agent
    my_agent = CNN_Agent(env_name, QNetwork_Class, max_iter_or_epis, dueling)

    # set the file path
    my_agent.set_file_path(file_path)

    # do training and evaluation
    if do_train:
        my_agent.train(replay_memory_on)
        my_agent.plot_evaluations()
    
#    # do testing
#    if env_name != "SpaceInvaders-v0":
#        my_agent.test()
#    my_agent.plot_test()
#
#    # analyze the testing result
#    test_rewards = np.array(my_agent.all_test_rewards)
#    test_mean = np.mean(test_rewards)
#    test_std = np.std(test_rewards)
#
#    print("final reward mean: {}; std: {}".format(test_mean, test_std))


if __name__ == '__main__':
    env_name = "SpaceInvaders-v0" # choose the environment
    file_path = "./" + env_name  # path to save the model, videos and plots

    QNetwork_Class = Deep_Q_CNN
    
    replay_memory_on = 1 # whether to use replay memory
    dueling = 0          # whether to use dueling architecture

    max_iter_or_epis = 10000000  # maximum training iterations or episodes
    do_train = 1         # whether to train the model

    main(env_name, file_path, QNetwork_Class, replay_memory_on,
         max_iter_or_epis, dueling, do_train)

