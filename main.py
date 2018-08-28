#!/usr/bin/env python3

import numpy as np
import env_wrapper
import matplotlib.pyplot as plt
from copy import deepcopy
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

#########################################################################
# INIT: Global Constants
#########################################################################

# env_name = 'MountainCar-v0'
env_name = 'CartPole-v0'
# env_name = "AND"
# env_name = "FrozenLakeNotSlippery-v0"
env_wrapper = env_wrapper.env_wrapper()
env = env_wrapper.make(env_name)
# NOTE: e.g. for n_obs read: number of observations; n_actions = number of actions

# contraption to retrieve observation space regardless of box type or discrete type. will not work if any other
# incompatible type is given. a method like env.observation_space.flattened that always gives an int would be nice @gym
n_obs = 0
try:
    n_obs = env.observation_space.high.__len__()
except:
    pass

if n_obs == 0:
    n_obs = env.observation_space.n

n_actions = env.action_space.n

# n_obs = 2
# n_actions = 3

fixed_seed = True  # specify a fixed seed if wanted. Chosen seed will be displayed.

use_seed_list = True  # Only works if fixed_seed is true, overwrites seed variable successivley

original_seed_list = [3465711357, 3764021189, 3401113854, 4159376773, 4023832405, 1945283214, 2160520317, 3704691305,
                      3906513104, 3579410646]

seed_list = deepcopy(original_seed_list)

seed = 0  # fixed seed to be set, if wanted. Value only used if fixed_seed and not use_seed_list


np.seterr(over='print')

# index variables for trajectory structure to improve readability of the code
idx_observations = 0
idx_action_probabilities = 1
idx_action_one_hots = 2
idx_list_of_rewards = 3
# after conversion:
idx_discounted_sums_of_rewards = 3
idx_int_action = 4
idx_acc_reward = 5
idx_amount_of_steps = 6

show_figures = False  # if set to false figures will be saved instead

# shuffled index for numpy to have high contrast colors for variables in plot
# idx_color = np.arange(len(list_of_colors))
# np.random.shuffle(idx_color)

#########################################################################
# INIT: Global Parameters and Variables
#########################################################################

#############
# General:

# values for initializing any new agent agent
initialization_mean = 0
initialization_std = 0.01
# maximum amount of steps allowed to take per played game
max_steps = 200
render_env = False

batch_size = 100

# Global for holding achieved rewards
mean_reward_holder = []
std_reward_holder = []



#############
# F0:

n_batches = 25
n_train_episodes = batch_size * n_batches
F0_threshold = 300  # reward threshold after which adaboost shall take over if reached prior to batch limit

learning_rate = .5
# discount factor gamma
gamma = 0.9999
baseline = 0.5  # a baseline between 0 and 1 seems to help a bit

rmsprop_factor_old = 0.9
rmsprop_factor_new = 0.1

#############
# Adaboost:

# NOTE: e.g. for n_agents read: number of agents
n_agents = 0  # number of adaboost agents

# Value by which favorable actions get encouraged/discouraged by.
probability_adjustment = 0.8

l2_regularization_lambda = 0.01

current_algorithm = ""

#########################################################################
# Function Definitions
#########################################################################

def softmax(input_array):
    input_raised_by_e = np.exp(input_array)
    return input_raised_by_e / np.sum(input_raised_by_e)

# calculate action probabilities
def get_action_probabilities(agent, states):
    return softmax(np.dot(states, agent[0]))

# get random action based on probabilities as integer
def get_random_choice(action_probabilities):
    # print(np.random.choice(np.arange(n_actions), p=action_probabilities.ravel()))
    # print(np.random.choice(np.arange(n_actions), p=action_probabilities.ravel()))

    return np.random.choice(np.arange(n_actions), p=action_probabilities.ravel())

# get one hot of random choice and the action as integer
def get_random_choice_one_hot(action_probabilities):
    one_hot = np.zeros([n_actions])
    action = get_random_choice(action_probabilities)
    one_hot[action] = 1.
    return one_hot, action


# get action based on highest probability. Take first action if probabilities are equal
def get_testing_choice(action_probabilities):
    return np.argmax(action_probabilities.ravel())


# returns one hot with taken action with highest probability and the action as integer
def get_testing_choice_one_hot(action_probabilities):
    one_hot = np.zeros([n_actions])
    action = get_testing_choice(action_probabilities)
    one_hot[action] = 1.
    return one_hot, action


# function handler for either testing or training modes
def get_choice_one_hot(action_probabilities, is_training):
    if is_training:
        return get_random_choice_one_hot(action_probabilities)
    else:
        return get_testing_choice_one_hot(action_probabilities)


# calculates discounted sum of reward for each timestep and adds it to given trajectory
# trajectory: [[[observation]],[[action_probabilities]], [[action_one_hot]],
#               [list_of_rewards],[action_as_integer], acc_reward, total_steps_taken]
# takes trajectory and replaces list of rewards with list of discounted future rewards and returns trajectory
def calculate_discounted_sums_of_reward(trajectory):
    n_steps = len(trajectory[idx_list_of_rewards])
    # print(trajectory)
    if n_steps > 1:
        for i in range(n_steps):
            discounted_sum_of_future_rewards = 0
            for k in range(i, n_steps):
                discounted_sum_of_future_rewards += (gamma ** (k - i)) * trajectory[idx_list_of_rewards][k]

            trajectory[idx_list_of_rewards][i] = discounted_sum_of_future_rewards

    return trajectory


# calculates gradient(of the loss) for a given set of trajectories
# param: trajectories: expects list of trajectories.
# trajectory: [[[observation]],[[action_probabilities]], [[action_one_hot]],
#               [discounted_sums_of_rewards],[action_as_integer], acc_reward, total_steps_taken]
def calculate_gradient(trajectories):
    # gradient has shape of network/theta
    gradient = np.zeros([n_obs + 1, n_actions])
    rmsprop_holder = np.zeros([n_obs + 1, n_actions])

    for trajectory in trajectories:
        # to be added to variable gradient when trajectory has been processed
        curr_gradient = np.zeros([n_obs + 1, n_actions])

        # do stuff
        n_steps = len(trajectory[idx_discounted_sums_of_rewards])
        for step in range(n_steps):
            # extract values of current step
            observation = trajectory[idx_observations][step]  # -> 4-dim vector
            action_probabilities = trajectory[idx_action_probabilities][step]  # -> 2-dim vector
            # probability of action taken at step
            action = trajectory[idx_int_action][step]  # -> scalar 0 or 1
            action_probability = action_probabilities[action]  # -> scalar

            action_one_hot = trajectory[idx_action_one_hots][step]  # -> 2-dim vector

            # print(observation.shape, action_probabilities.shape, action, action_probability, action_one_hot.shape)

            discounted_sum_of_reward = trajectory[idx_discounted_sums_of_rewards][step]
            reward = discounted_sum_of_reward - baseline

            grad_log_z = action_probabilities[0] * observation
            if n_actions > 1:
                for action_counter in range(1, n_actions):
                    grad_log_z = np.vstack([grad_log_z, action_probabilities[action_counter] * observation])

                grad_log_z = grad_log_z.T

            assert grad_log_z.shape == (n_obs + 1, n_actions), grad_log_z.shape


            # calculate different parts of the gradient
            for column in range(n_actions):
                obs = -grad_log_z
                # obs += observation * action_one_hot
                # reset obs vector (single value of state currently processed)
                for row in range(n_obs + 1):
                    # position of gradient = column, part of state = row
                    obs[row][column] += observation[row]

                    # gradient of loss for step = 1/probability of current action * reward * (observation * one_hot)
                    # curr_gradient[row][column] += 1./action_probability * np.matmul(obs, action_one_hot) * reward
                    curr_gradient[row][column] += (1. / action_probability) * np.sum(obs[row, :].T * action_one_hot) * reward

                    # update rmsprop
                    rmsprop_holder[row][column] = rmsprop_holder[row][column] * rmsprop_factor_old \
                                                  + rmsprop_factor_new * curr_gradient[row][column]**2
                    curr_gradient[row][column] /= rmsprop_holder[row][column] ** 0.5

        gradient += curr_gradient

    return gradient / len(trajectories)

# converts observation into an np array one hot vector or normal np array
def convert_observation(observation):
    if np.array(observation).shape == (n_obs, ):
        return np.hstack([1, np.array(observation)])

    # else:
    one_hot_obs = np.zeros([n_obs + 1])
    one_hot_obs[observation] = 1.
    one_hot_obs[0] = 1.
    return one_hot_obs

# returns trajectory generated by agent playing the game
# trajectory: [[[observation]],[[action_probabilities]], [[action_one_hot]],
#               [list_of_rewards],[action_as_integer], acc_reward, total_steps_taken]
def generate_batch_of_trajectories(agents, is_training):
    trajectories = []
    for episode in range(batch_size):
        step = 0
        done = False
        observation = convert_observation(env.reset())

        # render env?
        if render_env:
            env.render()

        observation_holder = [observation]
        action_probabilities_holder = []
        one_hot_holder = []
        list_of_rewards_holder = []
        action_holder = []
        acc_reward_holder = 0

        # start of episode
        while not done:
            step += 1

            # voting is done by adding probabilities together and taking the mean
            action_probabilities = np.zeros([n_actions])

            for agent in agents:
                # get action probabilities for given observation
                curr_action_probabilities = get_action_probabilities(agent, observation)
                action_probabilities += curr_action_probabilities

            action_probabilities /= len(agents)


            # get random action(bool is_training decides whether choice is random/if highest probable action is taken)
            action_one_hot, action = get_choice_one_hot(action_probabilities, is_training)


            # take the action
            observation, reward, done, info = env.step(action)

            # save all the values
            action_probabilities_holder.append(action_probabilities)
            one_hot_holder.append(action_one_hot)
            list_of_rewards_holder.append(reward)
            action_holder.append(action)
            acc_reward_holder += reward

            if step >= max_steps:
                done = True

            if not done:
                observation = convert_observation(observation)
                observation_holder.append(observation)

        # episode has ended: create and save trajectory
        trajectory = [observation_holder, action_probabilities_holder, one_hot_holder, list_of_rewards_holder,
                      action_holder, acc_reward_holder, step]
        # transform list of rewards to discounted sums of reward for whole trajectory
        trajectory = calculate_discounted_sums_of_reward(trajectory)
        # save trajectory
        trajectories.append(trajectory)

    return trajectories


# takes trajectories of a batch and returns of a list with length = longest trajectory containing average
# discounted reward for each step
def calculate_mean_of_sums_of_reward_by_step(trajectories):
    # list will be as long as the longest trajectory
    average_discounted_reward_per_step = []

    for step in range(max_steps):
        total_reward = 0
        instance_found = False
        for trajectory in range(len(trajectories)):
            if trajectories[trajectory][idx_amount_of_steps] - 1 >= step:
                instance_found = True
                total_reward += trajectories[trajectory][idx_discounted_sums_of_rewards][step]
                # else:
                # total_reward += 0

        # if any trajectory was as long as step
        if instance_found:
            average_reward_for_current_step = total_reward / len(trajectories)
            average_discounted_reward_per_step.append(average_reward_for_current_step)
        else:
            # no trajectory reached this length. we are done.
            break

    return average_discounted_reward_per_step


# takes trajectories and then updates the recorded action probabilities to targets to be used by the next new agent
def calculate_targets_for_adaboost_agents(trajectories):
    average_discounted_rewards = calculate_mean_of_sums_of_reward_by_step(trajectories)

    for trajectory in range(len(trajectories)):
        for step in range(len(trajectories[trajectory][idx_discounted_sums_of_rewards])):
            adjustment = probability_adjustment
            if trajectories[trajectory][idx_discounted_sums_of_rewards][step] <= average_discounted_rewards[step]:
                # if reward is lower than average, discourage this course of action
                adjustment *= -1
                # print("Discourage")
            else:
                # print("Encourage")
                pass

            action_taken = trajectories[trajectory][idx_int_action][step]
            trajectories[trajectory][idx_action_probabilities][step][action_taken] += adjustment


            one_hundred_percent = False
            if trajectories[trajectory][idx_action_probabilities][step][action_taken] >= 1:
                one_hundred_percent = True
                trajectories[trajectory][idx_action_probabilities][step][action_taken] = 1

            # change remaining possibilities so that sum of possibilities is 1 again
            compensation = adjustment / (n_actions - 1) * (-1)
            for action in range(n_actions):
                if action != action_taken:
                    if one_hundred_percent:
                        trajectories[trajectory][idx_action_probabilities][step][action] = 0
                    else:
                        trajectories[trajectory][idx_action_probabilities][step][action] += compensation


    return trajectories

# takes trajectories and then updates the recorded action probabilities to targets to be used by the next new agent
def calculate_targets_for_adaboost_agents_policyboost_style(trajectories):
    average_discounted_rewards = calculate_mean_of_sums_of_reward_by_step(trajectories)

    for trajectory in range(len(trajectories)):
        for step in range(len(trajectories[trajectory][idx_discounted_sums_of_rewards])):
            action_taken = trajectories[trajectory][idx_int_action][step]
            new_action_probabilities = np.zeros([n_actions])
            centered_reward = trajectories[trajectory][idx_discounted_sums_of_rewards][step] \
                        - average_discounted_rewards[step]

            for action in range(n_actions):
                if action == action_taken:
                    new_action_probabilities[action] = (trajectories[trajectory][idx_action_probabilities][step][action]\
                                            * (1 - trajectories[trajectory][idx_action_probabilities][step][action]))\
                                            * centered_reward
                else:
                    # NPPG & PolicyBoost mixed:
                    new_action_probabilities[action] = -trajectories[trajectory][idx_action_probabilities][step][action]\
                                            * trajectories[trajectory][idx_action_probabilities][step][action_taken]\
                                            * centered_reward
                    # for testing(this is the Policyboost style):
                    new_action_probabilities[action] = 0

            trajectories[trajectory][idx_action_probabilities][step] = new_action_probabilities

    return trajectories


# fits linear weak learner with analytic solution to linear regression
def train_adaboost_agent(targets):
    design_matrix = None
    y = None
    for target in targets:
        for step in range(target[idx_amount_of_steps]):
            if design_matrix is None:
                design_matrix = target[idx_observations][step]
                y = target[idx_action_probabilities][step]
            else:
                design_matrix = np.vstack([design_matrix, target[idx_observations][step]])
                y = np.vstack([y, target[idx_action_probabilities][step]])

    # since the agent gets created from the observations, where we already added the constant 1
    # we don't need to manually add a bias.

    agent = (np.dot(design_matrix.T, design_matrix))
    regularization_term = (np.identity(agent.shape[1]) * l2_regularization_lambda)
    agent = np.dot(np.dot(np.linalg.inv(agent + regularization_term), design_matrix.T), y)

    # print("Training MSE for current agent: " + str(np.sum((np.dot(design_matrix, agent) - y) ** 2, axis=0)))

    return [agent], design_matrix, y


def analyze_adaboost_agent_performance(agent, design_matrix, targets):
    outputs = np.dot(design_matrix, agent[0])
    length = outputs.shape[0]

    success_counter = 0.0

    # get difference between action probabilities since this will determine which action will be taken
    for step in range(length):
        # the difference of the action probabilities shows which action had the greater probability
        # if the product of diff_out and diff_target is positive, the same action was chosen
        if (outputs[step][0] - outputs[step][1]) * (targets[step][0] - targets[step][1]) > 0:
            # correct
            success_counter += 1

    # return success rate
    return success_counter/length



def analyze_reward_of_trajectory_batch(trajectories, is_training=True):
    list_of_acc_rewards = []
    for trajectory in trajectories:
        list_of_acc_rewards.append(trajectory[idx_acc_reward])

    list_of_acc_rewards = np.array(list_of_acc_rewards)
    mean = np.mean(list_of_acc_rewards)
    std = np.std(list_of_acc_rewards)

    if not is_training:
        mean_reward_holder.append(mean)
        std_reward_holder.append(std)

    return mean, std

# creates randomly initialized agent and trains it until better than random
def train_F0():

    theta = np.random.normal(initialization_mean, initialization_std, (n_obs, n_actions))
    # bias can be initialized 0, 1 or randomly. 1 seems to work best
    # bias = np.zeros([n_actions])
    bias = np.ones([n_actions])
    # bias = np.random.normal(initialization_mean, initialization_std, (n_actions))

    theta = np.vstack([bias, theta])

    # print(theta)

    agent = [theta]

    training_mean_holder = []
    training_std_holder = []

    is_training = False

    trajectories = generate_batch_of_trajectories([agent], is_training)
    mean, std = analyze_reward_of_trajectory_batch(trajectories, is_training)


    print("Average reward for random agent over " + str(batch_size) + " episodes: " + str(mean) + " +- " + str(std))

    print("Currently training agent 0")
    is_training = True


    for batch in range(n_batches):
        trajectories = generate_batch_of_trajectories([agent], is_training)

        # print results of batch
        mean, std = analyze_reward_of_trajectory_batch(trajectories)
        print("Average training reward after " + str((batch + 1) * batch_size) + " episodes: "
              + str(mean) + " +- " + str(std))

        training_mean_holder.append(mean)
        training_std_holder.append(std)

        # calculate gradient
        gradient = calculate_gradient(trajectories)
        # apply gradient to theta of agent
        agent[0] -= learning_rate * gradient


       # if batch % 5 == 0:
        trajectories = generate_batch_of_trajectories([agent], False)
        mean, std = analyze_reward_of_trajectory_batch(trajectories, n_agents != 0)  # False for Policy Gradient only
        print("Average testing reward after " + str((batch + 1) * batch_size) + " episodes: "
              + str(mean) + " +- " + str(std))

        if mean > F0_threshold:
            print("Average reward bigger than " + str(F0_threshold)
                  + ". Returning F0 to start with adaboost algorithm!\n")
            break

    return agent, training_mean_holder, training_std_holder

def plot_results(agent_test_results, list_of_fms, training_mean_holder, training_std_holder):
    save_figures = True
    if show_figures:
        save_figures = False


    # Overall Testing Performance Plot
    plt.figure(0)
    mean_holder = np.array(mean_reward_holder)
    std_holder = np.array(std_reward_holder)

    x = np.linspace(0, mean_holder.shape[0], mean_holder.shape[0])

    unit = 'epochs'
    if current_algorithm != 'PolicyGradient':
        unit = 'agents'

    plt.plot(x, mean_holder, 'k-')
    plt.xlabel('Number of '+unit+'\nSeed: ' + str(seed)) #  TODO: agents in network
    plt.ylabel('Average reward over 100 testing episodes')
    plt.title('Average testing reward of agent over 100 episodes\nby number of '+unit+' in ' + env_name)

    plt.fill_between(x, mean_holder - std_holder, mean_holder + std_holder)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.15)

    if save_figures:
        plt.savefig(current_algorithm + str(seed) + '_testing.png', format='png')
        plt.clf()

    # Individual Adaboost agent performance plot
    plt.figure(1)

    successes = np.array(agent_test_results)

    plt.title('Success rate on training data for individual ' + current_algorithm +' agents')
    plt.xlabel('Agent ID')
    plt.ylabel('Percentage')

    x = np.arange(len(agent_test_results))
    x += 1
    plt.plot(x, successes)
    plt.legend(['Success Rate'], loc='upper right')

    if save_figures:
        plt.savefig(current_algorithm + str(seed) + '_success_rate.png', format='png')
        plt.clf()

    # History of values of Theta
    plt.figure(2)
    values_of_theta = []


    for theta_i in range((n_obs+1) * n_actions):
        curr_list = []
        for Fm in list_of_fms:
            value = Fm[0][theta_i % (n_obs+1)][theta_i // (n_obs+1)]
            curr_list.append(value)
        values_of_theta.append(curr_list)

    x = np.arange(len(values_of_theta[0]))
    for i in range(len(values_of_theta)):
        plt.plot(x, np.array(values_of_theta[i]))

    plt.title('History of the values of $\Theta$')
    plt.xlabel('Number of '+unit)
    plt.ylabel('Real Value')

    string = []

    for i in range(n_actions):
        for x in range(n_obs+1):
            string.append('$\Theta_{' + str(i+1) + str(x+1) + '}$')

    plt.legend(string, loc='upper right')

    if save_figures:
        plt.savefig(current_algorithm + str(seed) + '_values_of_theta.png', format='png')
        plt.clf()

    # Training Performance
    plt.figure(3)

    x = np.arange(len(training_mean_holder))
    training_mean_holder = np.array(training_mean_holder)
    training_std_holder = np.array(training_std_holder)

    plt.plot(x, training_mean_holder, color='black')
    plt.fill_between(x, training_mean_holder - training_std_holder, training_mean_holder + training_std_holder)

    plt.title('Training performance per '+unit[:-1] )  # todo change back to agent ID
    plt.xlabel('Number of '+unit)
    plt.ylabel('Average reward over 100 training episodes')

    if save_figures:
        plt.savefig(current_algorithm + str(seed) + '_training.png', format='png')
        plt.clf()
    else:
        plt.show()

#########################################################################
# MAIN
#########################################################################

def main():
    # seed has to be a 32bit unsigned integer
    global seed
    global seed_list
    if not fixed_seed:
        seed = np.random.randint(low=0, high=42 * 10 ** 8, size=1)[0]
    elif use_seed_list:
        seed = seed_list[0]
        seed_list.pop(0)

    env.seed(seed)
    np.random.seed(seed)
    print("Current Setting:")
    print("Seed: " + str(seed))
    print("Number of Boosting Agents: " + str(n_agents))
    print("Number of Batches: " + str(n_batches))
    print("Batchsize: " + str(batch_size))
    print("F0 Threshold: " + str(F0_threshold))
    print("Current Algorithm: " + current_algorithm)

    training_mean_holder = []
    training_std_holder = []

    # initialize Fm with slightly trained F0
    print("Seed used for np.random and gym: " + str(seed))
    agent, mean, std = train_F0()

    if n_agents == 0:
        # if we only do gradient descent, lets see all its training episodes
        training_mean_holder += mean
        training_std_holder += std
    else:
        # if we do train other agents, lets only see the latest training results by gradient descent we start off with
        training_mean_holder = [mean[-1]]
        training_std_holder = [std[-1]]

    print("Length of list training mean holder after training F0: " + str(len(training_std_holder)))

    Fm = [agent]
    Fm_old = [deepcopy(agent)]

    is_training = False
    trajectories = generate_batch_of_trajectories(Fm, is_training)
    mean, std = analyze_reward_of_trajectory_batch(trajectories, is_training)

    print("Average reward for testing F0 over " + str(batch_size) + " episodes: " + str(mean) + " +- " + str(std))

    agent_test_results = []

    # start of boosting algorithm
    for curr_agent_id in range(1, n_agents + 1):
        print("Currently training agent " + str(curr_agent_id))
        # 1. get trajectories, one new batch per agent
        # 2. calculate all discounted sums of rewards
        is_training = True
        trajectories = generate_batch_of_trajectories(Fm, is_training)

        mean, std = analyze_reward_of_trajectory_batch(trajectories)
        training_mean_holder.append(mean)
        training_std_holder.append(std)

        targets = []

        if current_algorithm == "PolicyBoost":
            targets = calculate_targets_for_adaboost_agents_policyboost_style(trajectories)
        elif current_algorithm == "HeuristicPolicyBoost":
            targets = calculate_targets_for_adaboost_agents(trajectories)


        # 3a train new agent, giving it past choices and the corresponding rewards
        agent, design_matrix, targets = train_adaboost_agent(targets)

        agent_test_results.append(analyze_adaboost_agent_performance(agent, design_matrix, targets))

        # 3b find corresponding scaling parameter beta; skip for now!
        # 4. linearly combine new agent with old agent.
        # Fm.append(agent)
        Fm[0] += agent[0]  # * 1.0/25
        Fm_old.append(deepcopy(Fm[0]))
        # print(Fm[0])

        # render last batch of test rounds if wanted
        if curr_agent_id == n_agents:
            # render_env = True
            pass

        # test current network of agents
        is_training = False
        trajectories = generate_batch_of_trajectories(Fm, is_training)
        mean, std = analyze_reward_of_trajectory_batch(trajectories, is_training)
        print("Average reward for testing over " + str(batch_size) + " episodes: " + str(mean) + " +- " + str(std))

    #############
    # Plot
    plot_results(agent_test_results, Fm_old, training_mean_holder, training_std_holder)



if __name__ == '__main__':

    ##############################################################################################
    #### PolicyGradient run
    ##############################################################################################

    global current_algorithm
    current_algorithm = "PolicyGradient"

    num_its = len(seed_list)
    if show_figures:
        num_its = 1

    all_mean_rewards = []

    for i in range(num_its):
        print("--------------------")

        main()
        plt.close()
        # global mean_reward_holder
        # global std_reward_holder

        all_mean_rewards.append(deepcopy(mean_reward_holder))
        del mean_reward_holder[:]
        del std_reward_holder[:]



    # plot a line for each single run in same graph

    plt.figure(0)

    x = np.arange(len(all_mean_rewards[0]))
    for i in range(len(all_mean_rewards)):
        plt.plot(x, np.array(all_mean_rewards[i]))

    plt.title('Average Testing rewards over a set list of seeds in PolicyGradient')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Reward over 100 testing episodes')


    plt.savefig('Collected_graph_for_PolicyGradient.png', format='png')
    plt.clf()

    plt.close()

    # calculate overall average of all policygradient runs
    total_mean_PG = np.mean(np.array(all_mean_rewards), axis=0)
    total_averages_PG = np.mean(np.array(all_mean_rewards), axis=1)

    ##############################################################################################
    #### PolicyBoost run
    ##############################################################################################

    # do 2nd run with either adaboost enabled or disabled(just not the same way as before)

    global F0_threshold
    global n_batches
    global n_agents
    global seed_list

    F0_threshold = 80
    n_batches = 5
    n_agents = 25
    seed_list = deepcopy(original_seed_list)
    current_algorithm = "PolicyBoost"


    num_its = len(seed_list)
    if show_figures:
        num_its = 1

    all_mean_rewards = []

    for i in range(num_its):
        print("--------------------")

        main()
        plt.close()
        # global mean_reward_holder
        # global std_reward_holder

        all_mean_rewards.append(deepcopy(mean_reward_holder))
        del mean_reward_holder[:]
        del std_reward_holder[:]


    # plot a line for each single run in that graph

    plt.figure(0)

    x = np.arange(len(all_mean_rewards[0]))
    for i in range(len(all_mean_rewards)):
        plt.plot(x, np.array(all_mean_rewards[i]))

    plt.title('Average Testing rewards over a set list of seeds in PolicyBoost')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Reward over 100 testing episodes')

    plt.savefig('Collected_graph_for_PolicyBoost.png', format='png')
    plt.clf()

    plt.close()

    # calculate overall average of all policyboost runs
    total_mean_PB = np.mean(np.array(all_mean_rewards), axis=0)
    total_averages_PB = np.mean(np.array(all_mean_rewards), axis=1)


    ##############################################################################################
    #### HeuristicPolicyBoost run
    ##############################################################################################

    current_algorithm = "HeuristicPolicyBoost"
    seed_list = deepcopy(original_seed_list)


    num_its = len(seed_list)
    if show_figures:
        num_its = 1

    all_mean_rewards = []

    for i in range(num_its):
        print("--------------------")

        main()
        plt.close()
        # global mean_reward_holder
        # global std_reward_holder

        all_mean_rewards.append(deepcopy(mean_reward_holder))
        del mean_reward_holder[:]
        del std_reward_holder[:]


    # plot a line for each single run in that graph

    plt.figure(0)

    x = np.arange(len(all_mean_rewards[0]))
    for i in range(len(all_mean_rewards)):
        plt.plot(x, np.array(all_mean_rewards[i]))

    plt.title('Average Testing rewards over a set list of seeds in HeuristicPolicyBoost')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Reward over 100 testing episodes')

    plt.savefig('Collected_graph_for_HeuristicPolicyBoost.png', format='png')
    plt.clf()

    plt.close()

    # calculate overall average of all policyboost runs
    total_mean_HPB = np.mean(np.array(all_mean_rewards), axis=0)
    total_averages_HPB = np.mean(np.array(all_mean_rewards), axis=1)


    # plot both overall averages

    plt.figure(0)
    print("Total mean for each step PG:")
    print(total_mean_PG)

    print("Total mean for each step PB: ")
    print(total_mean_PB)

    print("Total mean for each step HPB: ")
    print(total_mean_HPB)

    print("Average over entire seed PG: ")
    print(total_averages_PG)

    print("Average over entire seed PB: ")
    print(total_averages_PB)

    print("Average over entire seed HPB: ")
    print(total_averages_HPB)

    print("Average over all seeds PG:")
    print(np.mean(total_averages_PG))

    print("Average over all seeds PB:")
    print(np.mean(total_averages_PB))

    print("Average over all seeds HPB:")
    print(np.mean(total_averages_HPB))

    if total_mean_PG.shape != total_mean_PB.shape:
        print("PG Shape: ")
        print(total_mean_PG.shape)
        print("PB Shape: ")
        print(total_mean_PB.shape)

    x = np.arange(len(all_mean_rewards[0]))
    plt.plot(x, total_mean_PG, label='PolicyGradient')
    plt.plot(x, total_mean_PB, label='PolicyBoost')
    plt.plot(x, total_mean_HPB, label='HeuristicPolicyBoost')

    plt.title('Average Testing rewards over a set list of seeds \n PolicyGradient vs. PolicyBoost vs. '
              'HeuristicPolicyBoost')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Reward over 100 testing episodes')
    plt.legend()

    plt.savefig('grand_total_Collected_graph_for_all_algorithms.png', format='png')
