import rnnCell

class RNN(object):

    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps

        if args.layerNormalization is True:

            if args.rnncell == 'rnn':
                self.cell_fn = rnnCell.lnBasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = lnGRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = lnBasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))
        else:
            if args.rnncell == 'rnn':
                self.cell_fn = BasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = tf.contrib.rnn.GRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = BasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))

        self.build_graph(args, maxTimeSteps)

    def build_graph(self, args, maxTimeSteps):

        self.graph = tf.Graph()

        with self.graph.as_default():

            self.inputX = tf.placeholder(tf.float32,
                                         shape=(maxTimeSteps, args.batch_size, args.num_feature))  # [maxL,32,39]
            inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])
            # self.inputList = tf.split(0, maxTimeSteps, inputXrs) #convert inputXrs from [32*maxL,39] to [32,maxL,39]
            
            self.inputList = tf.split(inputXrs, maxTimeSteps, 0)  # convert inputXrs from [32*maxL,39] to [32,maxL,39]

            self.targetIxs = tf.placeholder(tf.int64)
            self.targetVals = tf.placeholder(tf.int32)
            self.targetShape = tf.placeholder(tf.int64)
            self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)

            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))

            self.config = {'name': args.model,
                           'rnncell': self.cell_fn,
                           'num_hidden': args.num_hidden,
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'learning rate': args.learning_rate,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size}

            fbHrs = build_multi_dynamic_brnn(self.args, maxTimeSteps, self.inputX, self.cell_fn, self.seqLengths)

            with tf.name_scope('fc-layer'):

                with tf.variable_scope('fc'):
                    weightsClasses = tf.Variable(
                        tf.truncated_normal([args.num_hidden, args.num_class], name='weightsClasses'))
                    biasesClasses = tf.Variable(tf.zeros([args.num_class]), name='biasesClasses')
                    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in fbHrs]

            logits3d = tf.stack(logits)
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, logits3d, self.seqLengths))
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()

            if args.grad_clip == -1:
                # not apply gradient clipping
                self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            else:
                # apply gradient clipping
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op), args.grad_clip)
                opti = tf.train.AdamOptimizer(args.learning_rate)
                self.optimizer = opti.apply_gradients(zip(grads, self.var_trainable_op))
            self.predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(logits3d, self.seqLengths, merge_repeated=False)[0][0])
            if args.level == 'cha':
                self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
