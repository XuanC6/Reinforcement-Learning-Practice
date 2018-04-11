# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_func):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    policy = np.zeros(env.nS, dtype='int')  
            
    for s in range(env.nS):
        new_a = 0
        best_value = 0
        
        for a in range(env.nA):
            # get the info of next state
            left = 0
            right = 0
            possible_next_states = []
            is_terminal_list = []
            
            for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                left += prob*reward
                right += prob*value_func[nextstate]
                
                possible_next_states.append(nextstate)
                is_terminal_list.append(is_terminal)  
            # break if current state is terminal state
            if possible_next_states[0] == s and is_terminal_list[0]:
                break
            # calculate value of the state with action a
            v = left + gamma * right
            if v > best_value:
                best_value = v
                new_a = a    
                
        policy[s] = new_a

    return policy


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """   
    value_func = np.zeros(env.nS)
    n_iter = 0
    
    for _ in range(max_iterations):
        max_val_change = 0
        value_func_new = np.zeros(env.nS)
        
        for s in range(env.nS):
            old_v = value_func[s]
            a = policy[s]
            # get the info of next state
            left = 0
            right = 0
            possible_next_states = []
            is_terminal_list = []
            
            for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                left += prob*reward
                right += prob*value_func[nextstate]
                
                possible_next_states.append(nextstate)
                is_terminal_list.append(is_terminal)                
            # continue if current state is terminal state
            if possible_next_states[0] == s and is_terminal_list[0]:
                continue
            # update the value of next state
            value_func_new[s] = left + gamma * right     
            max_val_change = max(max_val_change, np.abs(old_v - value_func_new[s]))
            
        n_iter += 1
        value_func = value_func_new.copy()
        
        if max_val_change < tol:        
            break
    
    return value_func, n_iter


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)
    n_iter = 0
    
    for _ in range(max_iterations):
        max_val_change = 0
        
        for s in range(env.nS):
            old_v = value_func[s]
            a = policy[s]
            # get the info of next state
            left = 0
            right = 0
            possible_next_states = []
            is_terminal_list = []
            
            for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                left += prob*reward
                right += prob*value_func[nextstate]
                
                possible_next_states.append(nextstate)
                is_terminal_list.append(is_terminal)                
            # continue if current state is terminal state
            if possible_next_states[0] == s and is_terminal_list[0]:
                continue
            # update the value of next state
            value_func[s] = left + gamma * right            
            max_val_change = max(max_val_change, np.abs(old_v - value_func[s]))
            
        n_iter += 1
        
        if max_val_change < tol:        
            break
    
    return value_func, n_iter    


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)
    n_iter = 0
    
    for _ in range(max_iterations):
        max_val_change = 0
        
        # shuffle the states
        states = np.arange(env.nS)
        np.random.shuffle(states)
        
        for s in states:
            old_v = value_func[s]
            a = policy[s]
            # get the info of next state
            left = 0
            right = 0
            possible_next_states = []
            is_terminal_list = []
            
            for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                left += prob*reward
                right += prob*value_func[nextstate]
                
                possible_next_states.append(nextstate)
                is_terminal_list.append(is_terminal)                
            # continue if current state is terminal state
            if possible_next_states[0] == s and is_terminal_list[0]:
                continue
            # update the value of next state
            value_func[s] = left + gamma * right            
            max_val_change = max(max_val_change, np.abs(old_v - value_func[s]))
            
        n_iter += 1
        
        if max_val_change < tol:        
            break
    
    return value_func, n_iter 


def evaluate_policy_async_custom(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluate the value of a policy. Updates states by a student-defined
    heuristic. 

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    return np.zeros(env.nS), 0


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
        
    for s in range(env.nS):
        
        old_a = policy[s]
        new_a = 0
        best_value = 0
        
        for a in range(env.nA):
            # get the info of next state
            left = 0
            right = 0
            possible_next_states = []
            is_terminal_list = []
            
            for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                left += prob*reward
                right += prob*value_func[nextstate]
                
                possible_next_states.append(nextstate)
                is_terminal_list.append(is_terminal)  
            # break if current state is terminal state, nothing need improvement
            if possible_next_states[0] == s and is_terminal_list[0]:
                break
            # calculate value of the state with action a
            v = left + gamma * right
            if v > best_value:
                best_value = v
                new_a = a
                
        policy[s] = new_a
        if new_a != old_a:
            policy_stable = False

    return policy_stable, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')  
    value_func = np.zeros(env.nS)
    n_improvement = 0
    n_evaluation = 0
    
    while True:
        value_func, n_iter = evaluate_policy_sync(env, gamma, policy, max_iterations, tol)        
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        
        n_evaluation += n_iter
        n_improvement += 1
             
        if policy_stable:
            break
        
    return policy, value_func, n_improvement, n_evaluation


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """    
    policy = np.zeros(env.nS, dtype='int')  
    value_func = np.zeros(env.nS)
    n_improvement = 0
    n_evaluation = 0
    
    while True:
        value_func, n_iter = evaluate_policy_async_ordered(env, gamma, policy, max_iterations, tol)        
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        
        n_evaluation += n_iter
        n_improvement += 1
             
        if policy_stable:
            break
        
    return policy, value_func, n_improvement, n_evaluation    


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')  
    value_func = np.zeros(env.nS)
    n_improvement = 0
    n_evaluation = 0
    
    while True:
        value_func, n_iter = evaluate_policy_async_randperm(env, gamma, policy, max_iterations, tol)        
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        
        n_evaluation += n_iter
        n_improvement += 1
             
        if policy_stable:
            break
        
    return policy, value_func, n_improvement, n_evaluation   


def policy_iteration_async_custom(env, gamma, max_iterations=int(1e3),
                                  tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_custom methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    return policy, value_func, 0, 0


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    n_iter = 0
    
    for _ in range(max_iterations):
        max_val_change = 0
        value_func_new = np.zeros(env.nS)
        
        for s in range(env.nS):
            old_v = value_func[s]
            best_val = 0
            for a in range(env.nA):
                # get the info of next state
                left = 0
                right = 0
                possible_next_states = []
                is_terminal_list = []
                
                for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                    left += prob*reward
                    right += prob*value_func[nextstate]
                    
                    possible_next_states.append(nextstate)
                    is_terminal_list.append(is_terminal)  
                # break if current state is terminal state
                if possible_next_states[0] == s and is_terminal_list[0]:
                    break
                # update the value of next state
                v = left + gamma * right
                if v > best_val:
                    best_val = v              
            value_func_new[s] = best_val                
            max_val_change = max(max_val_change, np.abs(old_v - value_func_new[s]))
            
        n_iter += 1
        value_func = value_func_new.copy()
        
        if max_val_change < tol:        
            break
    
    return value_func, n_iter


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    n_iter = 0
    
    for _ in range(max_iterations):
        max_val_change = 0
        
        for s in range(env.nS):
            old_v = value_func[s]
            best_val = 0
            for a in range(env.nA):
                # get the info of next state
                left = 0
                right = 0
                possible_next_states = []
                is_terminal_list = []
                
                for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                    left += prob*reward
                    right += prob*value_func[nextstate]
                    
                    possible_next_states.append(nextstate)
                    is_terminal_list.append(is_terminal)  
                # break if current state is terminal state
                if possible_next_states[0] == s and is_terminal_list[0]:
                    break
                # update the value of next state
                v = left + gamma * right
                if v > best_val:
                    best_val = v              
            value_func[s] = best_val                
            max_val_change = max(max_val_change, np.abs(old_v - value_func[s]))
            
        n_iter += 1       
        if max_val_change < tol:        
            break
    
    return value_func, n_iter


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    n_iter = 0
    
    for _ in range(max_iterations):
        max_val_change = 0
        
        # shuffle the states
        states = np.arange(env.nS)
        np.random.shuffle(states)        
        
        for s in states:
            old_v = value_func[s]
            best_val = 0
            for a in range(env.nA):
                # get the info of next state
                left = 0
                right = 0
                possible_next_states = []
                is_terminal_list = []
                
                for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                    left += prob*reward
                    right += prob*value_func[nextstate]
                    
                    possible_next_states.append(nextstate)
                    is_terminal_list.append(is_terminal)  
                # break if current state is terminal state
                if possible_next_states[0] == s and is_terminal_list[0]:
                    break
                # update the value of next state
                v = left + gamma * right
                if v > best_val:
                    best_val = v              
            value_func[s] = best_val                
            max_val_change = max(max_val_change, np.abs(old_v - value_func[s]))
            
        n_iter += 1       
        if max_val_change < tol:        
            break
    
    return value_func, n_iter


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of Individual State Updates
    """
    value_func = np.zeros(env.nS)
    n_indi_sta_up = 0
    
    for _ in range(max_iterations):
        max_val_change = 0
        
        # kick off terminal states
        raw_states = np.arange(env.nS)
        states = []
        for s in raw_states:
            _, nextstate, _, is_terminal = env.P[s][0][0]
            if not (nextstate==s and is_terminal):
                states.append(s)
        states = np.asarray(states)
        
        for s in states:
            old_v = value_func[s]
            best_val = 0
            for a in range(env.nA):
                # get the info of next state
                left = 0
                right = 0                
                for (prob, nextstate, reward, is_terminal) in env.P[s][a]:
                    left += prob*reward
                    right += prob*value_func[nextstate]        
                # update the value of next state
                v = left + gamma * right
                if v > best_val:
                    best_val = v              
            value_func[s] = best_val                
            max_val_change = max(max_val_change, np.abs(old_v - value_func[s]))
            n_indi_sta_up += 1
            
      
        if max_val_change < tol:        
            break
    
    return value_func, n_indi_sta_up

