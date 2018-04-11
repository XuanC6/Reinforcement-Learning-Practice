import os
import sys
import gym
import keras
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.path = './LunarLander-v2_reinforce_0'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_path = self.path + '/saved_data.ckpt'
        
        self.save_interval = 500
        self.test_interval = 500
        
        # lists to store test results
        self.test_means = []
        self.test_stds = []
        self.test_episodes = []
        
        # Construnction Phase
        self.input = tf.placeholder(tf.float32, shape=[None, 8])
        output = self.model(self.input)
        
            # this is used for generate episode
        action_to_play = tf.multinomial(tf.log(output), 1)
        self.action_to_play = tf.reshape(action_to_play, [-1])
        
            # use one_hot to extract the score of the actions played
        self.actions = tf.placeholder(tf.int32, shape = (None))
        action_onehot = tf.one_hot(self.actions, 4)
        action_taken_score = tf.reduce_sum(output * action_onehot, axis = 1)
        
        self.return_val = tf.placeholder(tf.float32, shape = [None])
            # define the loss
        loss = -tf.reduce_mean(self.return_val * tf.log(action_taken_score))

            # set the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.train_step = optimizer.minimize(loss)
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def train(self, env, train_episodes, render, gamma = 1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        n_episodes = 0
        
        with tf.Session() as sess:
            
            if os.path.isfile(self.file_path + '.index'):
                self.saver.restore(sess, self.file_path)
                print('Data Restored')
                print(datetime.now())

            else:
                self.init.run()
                print('Data Initialized')
                print(datetime.now())
            
            while True:
                # 1. Generate an episode
                n_episodes += 1
                states, action_takens, rewards = self.generate_episode(env, render)
                rewards = rewards / 100
                # 2. Compute the returns G
                returns = Reinforce.compute_returns(rewards, gamma)
                # 3. Do an update
                self.train_step.run(feed_dict={
                        self.input: states, 
                        self.actions: action_takens, 
                        self.return_val: returns})
                
                if n_episodes % self.save_interval == 0:
                    self.saver.save(sess, self.file_path)
                    
                if n_episodes % self.test_interval == 0:
                    print(datetime.now())
                    self.test(env, n_episodes)
                
                if n_episodes >= train_episodes:
                    break
            
            print(datetime.now())
            print('training finished')
            self.saver.save(sess, self.file_path)

    def test(self, env, n_episodes, test_episodes = 100, render=False):
        # run 100 test episodes on current policy, 
        # recording the mean/std of the cumulative reward.
        total_rewards = []
        for i in range(test_episodes):
            _, _, rewards = self.generate_episode(env, render)
            total_rewards.append(sum(rewards))
            
        total_rewards = np.array(total_rewards)
        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)
        
        print('episodes completed:', n_episodes)
        print('test mean over {} episodes:'.format(test_episodes), reward_mean)
        print('test std:', reward_std)
        print('')
        
        self.test_episodes.append(n_episodes)
        self.test_means.append(reward_mean)
        self.test_stds.append(reward_std)

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []
        
        n_steps = 0

        obs = env.reset()
        if render:
            env.render()
        
        while True:
            states.append(obs)
            action = self.action_to_play.eval(feed_dict = \
                                            {self.input: obs.reshape((-1, 8))})
            
            n_steps += 1
            next_obs, reward, done, _ = env.step(action[0])
            if render:
                env.render()
            rewards.append(reward)
            actions.append(action[0])
            obs = next_obs
            
#            if done or n_steps >= 500:
            if done:
                break

        return np.array(states), np.array(actions), np.array(rewards)
    
    def plot_test_reward(self):
        plt.figure()
        
        plt.errorbar(self.test_episodes, self.test_means, yerr = self.test_stds)
        
        plt.title("total reward vs. the number of training episodes")
        plt.savefig(self.path + "/error_bars.png")
        plt.clf()

    @staticmethod
    def compute_returns(rewards, gamma):
        # compute the return G
        T = len(rewards)
        returns = np.zeros((T))
        return_G = 0
        
        for t in reversed(range(T)):
            return_G = rewards[t] + gamma * return_G
            returns[t] = return_G
            
        return returns


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-3, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    agent = Reinforce(model, lr)
    agent.train(env, num_episodes, render)
    agent.plot_test_reward()


if __name__ == '__main__':
    main(sys.argv)
