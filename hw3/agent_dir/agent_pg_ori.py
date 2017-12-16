#3529

from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import time

STOP_ACTION = 1
UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1, STOP_ACTION: 2}

class Agent_PG(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.env = env
        self.max_step = 10000
        self.learning_rate = 0.001
        self.input_shape = [None, 6400] #the observation
        self.dim_hidden = 200
        self.num_action = 6
        self.batch_size = 1
        self.batch_size_episodes = 1
        self.gamma = 0.99
        self.model_dir = "model_pg_cnn"

        self.value_scale = 1
        self.entropy_scale = 1
        self.model = args.model


        self._sess = tf.Session()

        self.saver = None
        self.input = tf.placeholder(tf.float32, shape=self.input_shape, name='X')
        self.up_probability = self.build_model(self.input)


        self.sampled_actions = tf.placeholder(tf.int32, (None,1), name='sampled_actions')
        self.discounted_reward = tf.placeholder(tf.float32, (None,1), name='discounted_reward')

        self.loss = tf.losses.log_loss(labels=self.sampled_actions, predictions=self.up_probability, weights=self.discounted_reward)

        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -0.3, 0.3), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs)



        #self.train_op = self.optimizer.minimize(self.loss)
        self.init_op = tf.global_variables_initializer()
        
        self.first_move = True
        self.last_observation = None
        self.current_observation = None


        if args.test_pg:
            #you can load your model here

            self.saver = tf.train.Saver()

            if self.model:
                self.saver.restore(self._sess, self.model)
                print('loading trained model')
            else:
                print("No trained moedel found")



    def build_model(self, X):

        observation = tf.reshape(X, [-1,6400])
        
        h = tf.layers.dense(observation, units=self.dim_hidden, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        p = tf.layers.dense(h, units=1, activation=tf.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return p

    def build_model_cnn(self, X):

        X = tf.reshape(X, [-1, 80, 80, 1])
        W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.0, shape=[16]))

        X = tf.nn.relu(tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.0, shape=[32]))

        conv_out = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        flat = tf.contrib.layers.flatten(conv_out)

        p = tf.layers.dense(flat, units=1, activation=tf.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return p

    def process_frame(self, frame):
        
        """ Atari specific preprocessing, consistent with DeepMind """
        frame = frame[35:195]
        frame = frame[::2, ::2, 0]
        frame[ frame == 144] = 0
        frame[frame == 109] = 0
        frame[frame != 0] = 1

        return frame.astype(np.float).ravel()
        
    
    def discount_rewards(self, rewards, discount_factor):

        discounted_rewards = np.zeros_like(rewards)
        for t in range(len(rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                discounted_reward_sum += rewards[k] * discount
                discount *= discount_factor
                if rewards[k] != 0:
                    
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
        f = open('log/log_dense.txt', 'w')
        self.saver = tf.train.Saver()

        if self.model:
            self.saver.restore(self._sess, self.model)
        else:
            self._sess.run(self.init_op)


        batch_state_action_reward_tuples = []
        episode_n = 1

        while True:
            print("Starting episode %d" % episode_n)

            episode_done = False
            episode_reward_sum = 0

            round_n = 1

            last_observation = self.env.reset()
            last_observation = self.process_frame(last_observation)
            action = self.env.action_space.sample() ##random sample action

            observation, _, _, _ = self.env.step(action)
            observation = self.process_frame(observation)
            n_steps = 1

            while not episode_done:

                observation_delta = observation - last_observation
                last_observation = observation
                
                    
                up_probability = self._sess.run(self.up_probability, feed_dict = {self.input:observation_delta.reshape([1, -1])})

                if np.random.uniform() < up_probability:
                    action = UP_ACTION
                else:
                    action = DOWN_ACTION

                observation, reward, episode_done, info = self.env.step(action)
                observation = self.process_frame(observation)
                episode_reward_sum += reward
                n_steps += 1

                tup = (observation_delta, action_dict[action], reward)
                batch_state_action_reward_tuples.append(tup)

                if reward != 0:
                    round_n += 1
                    n_steps = 0

            print("Episode %d finished after %d rounds" % (episode_n, round_n))

            # exponentially smoothed version of reward

            print("Reward total was %.3f" \
                % (episode_reward_sum))

            if episode_n % self.batch_size_episodes == 0:

                states, actions, rewards = zip(*batch_state_action_reward_tuples)
                rewards = self.discount_rewards(rewards, self.gamma)
                rewards -= np.mean(rewards)
                rewards /= np.std(rewards)
                batch_state_action_reward_tuples = list(zip(states, actions, rewards))

                print("Training with %d (state, action, reward) tuples" % len(batch_state_action_reward_tuples))

                states, actions, rewards = zip(*batch_state_action_reward_tuples)
                states = np.vstack(states)
                actions = np.vstack(actions)
                rewards = np.vstack(rewards)
                
                feed_dict = {
                    self.input: states,
                    self.sampled_actions: actions,
                    self.discounted_reward: rewards
                }
                self._sess.run(self.train_op, feed_dict)

                batch_state_action_reward_tuples = []


            if episode_n % 100 ==0:
                self.saver.save(self._sess, "./%s/%s-%d.ckpt" % (self.model_dir,self.model_dir, episode_n))

            episode_n += 1
            s= str(episode_n) + '\t' + str(episode_reward_sum) + '\n'
            f.write(s)


    
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
        
            self.last_observation = self.process_frame(observation)
            action = self.env.action_space.sample() ##random sample action
            self.first_move = False
        else:

            self.observation = self.process_frame(observation)


            observation_delta = self.observation - self.last_observation
            self. last_observation = self.observation
                
                    
            up_probability = self._sess.run(self.up_probability, feed_dict = {self.input:observation_delta.reshape([1, -1])})

            if np.random.uniform() < up_probability:
                action = UP_ACTION
            else:
                action = DOWN_ACTION

        return action


