from agent_dir.agent import Agent
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np 
from keras.layers.advanced_activations import LeakyReLU  


FINAL_EXPLORATION = 0.05
TARGET_UPDATE = 1000
ONLINE_UPDATE = 4

MEMORY_SIZE = 10000
EXPLORATION = 1000000

START_EXPLORATION = 1.
TRAIN_START = 10000
LEARNING_RATE = 0.0001
DISCOUNT = 0.99

def lrelu(x, alpha = 0.01):
    return tf.maximum(x, alpha * x)

def clipped_error(x): 
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

class Agent_DDQN(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DDQN,self).__init__(env)

        self.num_action = 3
        self.minibatch = 32
        self.esp = 1
        self.model_path = "save/Breakout.ckpt"
        self.replay_memory = deque()

        
        self.input = tf.placeholder("float", [None, 84, 84, 4])

        self.f1 = tf.get_variable("f1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2 = tf.get_variable("f2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3 = tf.get_variable("f3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1 = tf.get_variable("w1", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2 = tf.get_variable("w2", shape=[512, self.num_action], initializer=tf.contrib.layers.xavier_initializer())

        self.py_x = self.build_model(self.input, self.f1, self.f2, self.f3 , self.w1, self.w2)


        self.f1_r = tf.get_variable("f1_r", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f2_r = tf.get_variable("f2_r", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.f3_r = tf.get_variable("f3_r", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        self.w1_r = tf.get_variable("w1_r", shape=[7*7*64,512], initializer=tf.contrib.layers.xavier_initializer())
        self.w2_r = tf.get_variable("w2_r", shape=[512, self.num_action], initializer=tf.contrib.layers.xavier_initializer())

        self.py_x_r =self.build_model(self.input, self.f1_r, self.f2_r,self.f3_r, self.w1_r, self.w2_r)


        self.rlist=[0]
        self.recent_rlist=[0]

        self.episode = 0

        self.epoch_score = deque()
        self.epoch_Q = deque()
        self.epoch_on = False
        self.average_Q = deque()
        self.average_reward = deque()
        self.no_life_game = False

        self.a= tf.placeholder(tf.int64, [None])
        self.y = tf.placeholder(tf.float32, [None])
        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')
        
        a_one_hot = tf.one_hot(self.a, self.num_action, 1.0, 0.0)
        self.q_value = tf.reduce_sum(tf.multiply(self.py_x, a_one_hot), reduction_indices=1)
        
        difference = self.q_target - self.q_value
        
        difference = tf.clip_by_value(difference, 0.0, 1)

        errors = clipped_error(difference)
        
        
        self.loss = tf.reduce_mean(tf.reduce_sum(errors))
        self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE,momentum=0,epsilon= 1e-8, decay=0)
        self.train_op = self.optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=None)

        cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
        self.sess = tf.Session(config=cfg)

        if args.test_dqn:
            #you can load your model here
            self.saver.restore(self.sess, save_path = './save/Breakout_ddqn.ckpt-0')
            print('loading trained model')

    def build_model(self, input1, f1, f2, f3, w1, w2):
    
        c1 = tf.nn.relu(tf.nn.conv2d(input1, f1, strides=[1, 4, 4, 1],data_format="NHWC", padding = "VALID"))
        c2 = tf.nn.relu(tf.nn.conv2d(c1, f2, strides=[1, 2, 2, 1],data_format="NHWC", padding="VALID"))
        c3 = tf.nn.relu(tf.nn.conv2d(c2, f3, strides=[1,1,1,1],data_format="NHWC", padding="VALID"))

        l1 = tf.reshape(c3, [-1, w1.get_shape().as_list()[0]])


        #l2 = tf.nn.relu(tf.matmul(l1, w1))
        l2 = lrelu(tf.matmul(l1, w1))
        pyx = tf.matmul(l2, w2)

        return pyx    

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass


    def train(self):
        """
        Implement your training algorithm here
        """

        f = open('reward.txt', 'w')
        epoch_on = False
        frame = 0
        #self.esp = 0.05
        self.sess.run(tf.global_variables_initializer())
        #self.saver.restore(self.sess, save_path = './save/Breakout.ckpt-0')
        self.sess.run(self.w1_r.assign(self.w1))
        self.sess.run(self.w2_r.assign(self.w2))
        self.sess.run(self.f1_r.assign(self.f1))
        self.sess.run(self.f2_r.assign(self.f2))
        self.sess.run(self.f3_r.assign(self.f3))

        while np.mean(self.recent_rlist) < 500 :
            self.episode += 1

            if len(self.recent_rlist) > 100:
                del self.recent_rlist[0]

            rall = 0
            d = False
            ter = False
            count = 0
            s = self.env.reset()
            avg_max_Q = 0
            avg_loss = 0
            epoch = 0 

            while not d :

                frame +=1
                count+=1

                if self.esp > FINAL_EXPLORATION and frame > TRAIN_START:
                    self.esp -= 4*(START_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION

                state = np.reshape(s, (1, 84, 84, 4))
                Q = self.sess.run(self.py_x, feed_dict = {self.input : state})

                self.average_Q.append(np.max(Q))
                avg_max_Q += np.max(Q)

                if self.esp > np.random.rand(1):
                    action = np.random.randint(self.num_action)
                else:
                    action = np.argmax(Q)

                if action == 0:
                    real_a = 1
                elif action == 1:
                    real_a = 2
                else:
                    real_a = 3

                s1, r, d, l = self.env.step(real_a)

                ter = d

                self.replay_memory.append((np.copy(s), np.copy(s1), action ,r, ter))     
                
                s = s1

                if len(self.replay_memory) > MEMORY_SIZE:
                    self.replay_memory.popleft()

                rall += r

                if frame > TRAIN_START and frame % ONLINE_UPDATE == 0:
                    s_stack = deque()
                    a_stack = deque()
                    r_stack = deque()
                    s1_stack = deque()
                    d_stack = deque()
                    y_stack = deque()

                    sample = random.sample(self.replay_memory, self.minibatch)

                    for _s , s_r, a_r, r_r, d_r in sample:
                        s_stack.append(_s)
                        a_stack.append(a_r)
                        r_stack.append(r_r)
                        s1_stack.append(s_r)
                        d_stack.append(d_r)

                    d_stack = np.array(d_stack) + 0

                    Q1 = self.sess.run( self.py_x_r, feed_dict={self.input: np.array(s1_stack)})
                    Q_online = self.sess.run( self.py_x, feed_dict={self.input: np.array(s1_stack)})

                    batch_index = np.arange(self.minibatch, dtype=np.int32)
                    max_act4next = np.argmax(Q_online, axis=1)        # the action that brings the highest value is evaluated by q_eval
                    

                    #expected_state_action_values = r_stack + DISCOUNT * np.max(Q1, axis=1)#max_act4next
                    expected_state_action_values = r_stack + DISCOUNT * max_act4next
                    
                    #selected_q_next = Q1[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
                    #expected_state_action_values
                    #y_stack = r_stack + (1 - d_stack) * DISCOUNT * selected_q_next

                    #self.sess.run(self.train_op, feed_dict={self.input: np.array(s_stack), self.y: y_stack, self.a: a_stack})
                    self.sess.run(self.train_op,feed_dict={self.input: np.array(s_stack),self.a:a_stack, self.q_target: expected_state_action_values})
                    
                    if frame % TARGET_UPDATE == 0 :
                        self.sess.run(self.w1_r.assign(self.w1))
                        self.sess.run(self.w2_r.assign(self.w2))
                        self.sess.run(self.f1_r.assign(self.f1))
                        self.sess.run(self.f2_r.assign(self.f2))
                        self.sess.run(self.f3_r.assign(self.f3))

                if (frame - TRAIN_START) % 50000 == 0:
                    epoch_on = True

            if epoch_on:

                epoch += 1
                self.epoch_score.append(np.mean(self.average_reward))
                self.epoch_Q.append(np.mean(self.average_Q))

                epoch_on = False
                average_reward = deque()
                average_Q = deque()

                save_path = self.saver.save(self.sess, self.model_path, global_step=(epoch-1))
                print("Model(episode :",self.episode, ") saved in file: ", save_path )

            self.recent_rlist.append(rall)
            self.rlist.append(rall)
            self.average_reward.append(rall)

            if self.episode % 10 == 0:
                print("Episode:{0:6d} | Frames:{1:9d} | Steps:{2:5d} | Reward:{3:3.0f} | e-greedy:{4:.5f} | Avg_Max_Q:{5:2.5f} | "
                    "Recent reward:{6:.5f}  ".format(self.episode, frame, count, rall, self.esp, avg_max_Q/float(count),np.mean(self.recent_rlist)))
                
                s = "%d\t%lf\n" % (self.episode, np.mean(self.recent_rlist))
                f.write(s)


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        state = np.reshape(observation, (1, 84, 84, 4))
        Q = self.sess.run(self.py_x, feed_dict = {self.input : state})
        
        action = np.argmax(Q)

        if action == 0:
            real_a = 1
        elif action == 1:
            real_a = 2
        else:
            real_a = 3

        return real_a

