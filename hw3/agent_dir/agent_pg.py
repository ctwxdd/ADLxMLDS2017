from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import time
import scipy
import threading
import os
import gym
import random

random.seed(87)

def copy_src_to_dst(from_scope, to_scope):

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def pipeline(image, new_HW=(80, 80), height_range=(35, 193), bg=(144, 72, 17)):

    image = crop_image(image, height_range)
    image = resize_image(image, new_HW)
    image = kill_background_grayscale(image, bg)
    image = np.expand_dims(image, axis=2)

    return image

def resize_image(image, new_HW):

    return scipy.misc.imresize(image, new_HW, interp="nearest")

def crop_image(image, height_range=(35, 195)):

    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]


def kill_background_grayscale(image, bg):

    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image


def discount_reward(rewards, gamma=0.99):

    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim, logdir=None):

        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
            net = self.states

            with tf.variable_scope("layer1"):
                net = tf.layers.conv2d(net,
                                       filters=16,
                                       kernel_size=(8, 8),
                                       strides=(4, 4),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("layer2"):
                net = tf.layers.conv2d(net,
                                       filters=32,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("fc1"):
                net = tf.contrib.layers.flatten(net)
                net = tf.layers.dense(net, 256, name='dense')
                net = tf.nn.relu(net, name='relu')

            # actor network
            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.action_prob = tf.nn.softmax(actions, name="action_prob")
            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            
            maximize_objective = log_action_prob * self.advantage + entropy * 0.005
            self.actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
            self.value_loss = tf.losses.mean_squared_error(labels=self.rewards,
                                                           predictions=self.values)

            self.total_loss = self.actor_loss + self.value_loss * .5
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=.99)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)

class Agent_worker(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, logdir=None):

        super(Agent_worker, self).__init__()
        self.local = A3CNetwork(name, input_shape, output_dim, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir
        self.episode = 0
        self.reward_log = open('log/'+self.name+'_log.txt', 'a')
        
        if self.name == 'thread_0':
            self.f = open('log/log.txt', 'w')

        self.state_diff = None

    def print(self, reward):
        message = "Agent(name={}, reward={})".format(self.name, reward)
        print(message)
        s= str(self.episode) + '\t' + str(reward) + '\n'
        self.reward_log.write(s)

    def play_episode(self):
        self.episode+=1
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []

        s = self.env.reset()
        s = pipeline(s)
        state_diff = s

        done = False
        total_reward = 0
        time_step = 0
        while not done:

            a = self.choose_action(state_diff)
            s2, r, done, _ = self.env.step(a)

            s2 = pipeline(s2)
            total_reward += r

            states.append(state_diff)
            actions.append(a)
            rewards.append(r)

            state_diff = s2 - s
            s = s2

            if r == -1 or r == 1 or done:
                time_step += 1

                if time_step >= 5 or done:
                    self.train(states, actions, rewards)
                    self.sess.run(self.global_to_local)
                    states, actions, rewards = [], [], []
                    time_step = 0

        self.print(total_reward)

    def run(self):
        try:
            while not self.coord.should_stop():
                self.play_episode()

        except Exception as e:
            self.coord.request_stop(e)

    def choose_action(self, states):
        """
        Args:
            states (2-D array): (N, H, W, 1)
        """
        states = np.reshape(states, [-1, *self.input_shape])
        feed = {
            self.local.states: states
        }

        action = self.sess.run(self.local.action_prob, feed)
        action = np.squeeze(action)

        return np.random.choice(np.arange(self.output_dim) + 1, p=action)

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions) - 1
        rewards = np.array(rewards)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = discount_reward(rewards, gamma=0.99)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards) + 1e-8

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients = self.sess.run(self.local.gradients, feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)

        
class Agent_PG(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)
        


        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.coord = tf.train.Coordinator()

        self.save_path = "model/pg/model.ckpt"
        self.n_threads = 8
        self.input_shape = [80, 80, 1]
        self.output_dim = 3  # {1, 2, 3}
        self.global_network = A3CNetwork(name="global",
                                    input_shape=self.input_shape,
                                    output_dim=self.output_dim)

        self.thread_list = []
        self.env_list = []

        if args.test_pg:
            #you can load your model here

            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.save_path)
        


    def _apply_discount(self, rewards, discount_factor):

        discounted_rewards = np.zeros_like(rewards)
        for t in range(len(rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                discounted_reward_sum += rewards[k] * discount
                discount *= discount_factor
                if rewards[k] != 0:
                    # Don't count rewards from subsequent rounds
                    break
            discounted_rewards[t] = discounted_reward_sum

        return discounted_rewards
 
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.first_move = True

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        try:
            for id in range(self.n_threads):
                print(id)
                env = gym.make("Pong-v0")

                single_agent = Agent_worker(env=env,
                                    session=self.sess,
                                    coord=self.coord,
                                    name="thread_{}".format(id),
                                    global_network=self.global_network,
                                    input_shape=self.input_shape,
                                    output_dim=self.output_dim)
                self.thread_list.append(single_agent)
                self.env_list.append(env)

            if tf.train.get_checkpoint_state(os.path.dirname(self.save_path)):
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
                saver = tf.train.Saver(var_list=var_list)
                saver.restore(self.sess, self.save_path)
                print("Model restored to global")
            else:
                init = tf.global_variables_initializer()
                self.sess.run(init)
                print("No model is found")

            for t in self.thread_list:
                t.start()

            print("Ctrl + C to close")


            self.coord.join(self.thread_list)
        
        except (KeyboardInterrupt, SystemExit):

            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
            saver = tf.train.Saver(var_list=var_list)
            saver.save(self.sess, self.save_path)
            print()
            print("=" * 10)
            print('Checkpoint Saved to {}'.format(self.save_path))
            print("=" * 10)

            print("Closing threads")
            self.coord.request_stop()
            self.coord.join(self.thread_list)

            print("Closing environments")
            for env in self.env_list:
                env.close()

            self.sess.close()


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        if (self.first_move):
        
            self.last_observation = pipeline(observation)
            action = self.env.action_space.sample() ##random sample action
            self.first_move = False

        else:
            curr_state = pipeline(observation)
            state_diff = curr_state - self.last_observation
            self.last_observation = curr_state

            states = np.reshape(state_diff, [-1, *self.input_shape])

            feed = {
                self.global_network.states: states
            }

            action = self.sess.run(self.global_network.action_prob, feed)
            action = np.squeeze(action)
            action = np.argmax(action) + 1
            #action = np.random.choice(np.arange(self.output_dim) + 1, p=action)
            
        return action



