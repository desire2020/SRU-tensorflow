import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn import RNNCell

class SRUCell(RNNCell):
    def __init__(self, num_units, activation=tf.nn.tanh, reuse=None):
        super(SRUCell, self).__init__(_reuse=reuse)
        self.hidden_dim = num_units
        self.g = activation
        init_matrix = tf.orthogonal_initializer()

        self.Wr = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]))
        self.br = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.U = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]))

    @property
    def state_size(self):
        return self.hidden_dim

    @property
    def output_size(self):
        return self.hidden_dim

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):
            c_prev = state

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(inputs, self.Wf) + self.bf
            )

            # Reset Gate
            r = tf.sigmoid(
                tf.matmul(inputs, self.Wr) + self.br
            )

            # Final Memory cell
            c = f * c_prev + (1.0 - f) * tf.matmul(inputs, self.U)

            # Current Hidden state
            current_hidden_state = r * self.g(c) + (1.0 - r) * inputs

            return current_hidden_state, c

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)