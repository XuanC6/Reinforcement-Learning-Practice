import os
import sys
import gym
import keras
import argparse
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from datetime import datetime

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

from reinforce import Reinforce

def critic():
    
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim = 8))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    return model


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n
        
        self.path = './LunarLander-v2_a2c'
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
        self.actions = tf.placeholder(tf.int32, shape = [None])
        action_onehot = tf.one_hot(self.actions, 4)
        action_taken_score = tf.reduce_sum(output * action_onehot,
                                           axis = 1)
        self.return_val = tf.placeholder(tf.float32, shape = [None])

            # critic value
        self.critic_val = self.critic_model(self.input)
            # adjust the dimension
        self.critic_val = tf.reshape(self.critic_val, [-1])
        self.critic_baseline = tf.placeholder(tf.float32, shape = [None])
        
            # define the loss
        loss_actor = -tf.reduce_mean((self.return_val-self.critic_baseline) * \
                                     tf.log(action_taken_score))
        loss_critic = tf.reduce_mean(tf.square(self.return_val-self.critic_val))

            # set the optimizer
        optimizer_actor = tf.train.AdamOptimizer(learning_rate = lr)
        optimizer_critic = tf.train.AdamOptimizer(learning_rate = critic_lr)

            # set the training step
        self.train_step_actor = optimizer_actor.minimize(loss_actor)
        self.train_step_critic = optimizer_critic.minimize(loss_critic)
        
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


    def train(self, env, train_episodes, render, gamma = 1):
        
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
                states, action_takens, rewards =\
                                       self.generate_episode(env, render)
                rewards = rewards / 100
                # 2. Compute the returns G(N_step)
                returns = self.compute_returns_N(states, rewards, gamma)
                
                critic_base = self.critic_val.eval(feed_dict={self.input: states})
                # 3. Do an update
                self.train_step_actor.run(feed_dict={
                        self.input: states,
                        self.actions: action_takens, 
                        self.return_val: returns,
                        self.critic_baseline: critic_base})
                
                self.train_step_critic.run(feed_dict={
                        self.input: states,
                        self.return_val: returns})
                
                # Save data or do test
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


    def compute_returns_N(self, states, rewards, gamma):
        # compute the return G(N_step)
        T = len(rewards)
        returns = np.zeros((T))
        
        for t in reversed(range(T)):
            
            if t + self.n >= T:
                Vend = 0
            else:
                state = states[t + self.n]
                Vend = self.critic_val.eval(feed_dict = \
                                            {self.input: state.reshape((-1,8))})
                Vend = Vend[0]
            
            signal = 0
            for k in range(self.n):
                if t+k < T:
                    reward = rewards[t+k]
                else:
                    reward = 0
                signal += (gamma**k) * reward

            returns[t] = signal + (gamma**self.n) * Vend
            
        return returns


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=5e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

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
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())
    
    critic_model = critic()

    agent = A2C(model, lr, critic_model, critic_lr, n)
    agent.train(env, num_episodes, render)
    agent.plot_test_reward()


if __name__ == '__main__':
    main(sys.argv)
