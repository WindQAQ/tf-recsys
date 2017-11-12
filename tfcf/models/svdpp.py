import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils

from .svd import SVD
from ..utils.data_utils import BatchGenerator
from ..metrics import mae
from ..metrics import rmse


def _convert_to_sparse_format(x):
    """Converts a list of lists into sparse format.  

    Args:
        x: A list of lists.

    Returns:
        A dictionary that contains three fields, which are 
            indices, values, and the dense shape of sparse matrix.
    """

    sparse = {
        'indices': [],
        'values': []
    }

    for row, x_i in enumerate(x):
        for col, x_ij in enumerate(x_i):
            sparse['indices'].append((row, col))
            sparse['values'].append(x_ij)

    max_col = np.max([len(x_i) for x_i in x]).astype(np.int32)

    sparse['dense_shape'] = (len(x), max_col)

    return sparse


def _get_implicit_feedback(x, num_users, num_items, dual):
    """Gets implicit feedback from (users, items) pair.

    Args:
        x: A numpy array of shape `(samples, 2)`.
        num_users: An integer, total number of users.
        num_items: An integer, total number of items.
        dual: A bool, deciding whether returns the
            dual term of implicit feedback of items.

    Returns:
        A dictionary that is the sparse format of implicit
            feedback of users, if dual is true.
        A tuple of dictionarys that are the sparse format of
            implicit feedback of users and items, otherwise.
    """

    if not dual:
        N = [[] for u in range(num_users)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)

        return _convert_to_sparse_format(N)
    else:
        N = [[] for u in range(num_users)]
        H = [[] for u in range(num_items)]
        for u, i, in zip(x[:, 0], x[:, 1]):
            N[u].append(i)
            H[i].append(u)

        return _convert_to_sparse_format(N), _convert_to_sparse_format(H)


class SVDPP(SVD):
    """Collaborative filtering model based on SVD++ algorithm.
    """

    def __init__(self, config, sess, dual=False):
        super(SVDPP, self).__init__(config, sess)
        self.dual = dual

    def _create_implicit_feedback(self, implicit_feedback, dual=False):
        """Returns the (tuple of) sparse tensor(s) of implicit feedback.
        """
        with tf.variable_scope('implicit_feedback'):
            if not dual:
                N = tf.SparseTensor(**implicit_feedback)

                return N
            else:
                N = tf.SparseTensor(**implicit_feedback[0])
                H = tf.SparseTensor(**implicit_feedback[1])

                return N, H

    def _create_user_terms(self, users, N):
        num_users = self.num_users
        num_items = self.num_items
        num_factors = self.num_factors

        p_u, b_u = super(SVDPP, self)._create_user_terms(users)

        with tf.variable_scope('user'):
            implicit_feedback_embeddings = tf.get_variable(
                name='implict_feedback_embedding',
                shape=[num_items, num_factors],
                initializer=tf.zeros_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_y_u))

            y_u = tf.gather(
                tf.nn.embedding_lookup_sparse(
                    implicit_feedback_embeddings,
                    N,
                    sp_weights=None,
                    combiner='sqrtn'),
                users,
                name='y_u'
            )

        return p_u, b_u, y_u

    def _create_item_terms(self, items, H=None):
        num_users = self.num_users
        num_items = self.num_items
        num_factors = self.num_factors

        q_i, b_i = super(SVDPP, self)._create_item_terms(items)

        if H is None:
            return q_i, b_i
        else:
            with tf.variable_scope('item'):
                implicit_feedback_embeddings = tf.get_variable(
                    name='implict_feedback_embedding',
                    shape=[num_users, num_factors],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.reg_g_i))

                g_i = tf.gather(
                    tf.nn.embedding_lookup_sparse(
                        implicit_feedback_embeddings,
                        H,
                        sp_weights=None,
                        combiner='sqrtn'),
                    items,
                    name='g_i'
                )

            return q_i, b_i, g_i

    def _create_prediction(self, mu, b_u, b_i, p_u, q_i, y_u, g_i=None):
        with tf.variable_scope('prediction'):
            if g_i is None:
                pred = tf.reduce_sum(
                    tf.multiply(tf.add(p_u, y_u), q_i),
                    axis=1)
            else:
                pred = tf.reduce_sum(
                    tf.multiply(tf.add(p_u, y_u), tf.add(q_i, g_i)),
                    axis=1)

            pred = tf.add_n([b_u, b_i, pred])

            pred = tf.add(pred, mu, name='pred')

        return pred

    def _build_graph(self, mu, implicit_feedback):
        _mu = super(SVDPP, self)._create_constants(mu)

        self._users, self._items, self._ratings = super(
            SVDPP, self)._create_placeholders()

        if not self.dual:
            N = self._create_implicit_feedback(implicit_feedback)

            p_u, b_u, y_u = self._create_user_terms(self._users, N)
            q_i, b_i = self._create_item_terms(self._items)

            self._pred = self._create_prediction(_mu, b_u, b_i, p_u, q_i, y_u)
        else:
            N, H = self._create_implicit_feedback(implicit_feedback, True)

            p_u, b_u, y_u = self._create_user_terms(self._users, N)
            q_i, b_i, g_i = self._create_item_terms(self._items, H)

            self._pred = self._create_prediction(
                _mu, b_u, b_i, p_u, q_i, y_u, g_i)

        loss = super(SVDPP, self)._create_loss(self._ratings, self._pred)

        self._optimizer = super(SVDPP, self)._create_optimizer(loss)

        self._built = True

    def train(self, x, y, epochs=100, batch_size=1024, validation_data=None):

        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        if not self._built:
            implicit_feedback = _get_implicit_feedback(
                x, self.num_users, self.num_items, self.dual)
            self._build_graph(np.mean(y), implicit_feedback)

        self._run_train(x, y, epochs, batch_size, validation_data)
