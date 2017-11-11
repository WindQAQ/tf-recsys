import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras import utils
except:
    from tensorflow.contrib.keras import utils

from .model_base import BaseModel
from ..utils.data_utils import BatchGenerator
from ..metrics import mae
from ..metrics import rmse


class SVD(BaseModel):
    """Collaborative filtering model based on SVD algorithm.
    """

    def __init__(self, config, sess):
        super(SVD, self).__init__(config)
        self._sess = sess

    def _create_placeholders(self):
        """Returns the placeholders.
        """
        with tf.variable_scope('placeholder'):
            users = tf.placeholder(tf.int32, shape=[None, ], name='users')
            items = tf.placeholder(tf.int32, shape=[None, ], name='items')
            ratings = tf.placeholder(
                tf.float32, shape=[None, ], name='ratings')

        return users, items, ratings

    def _create_constants(self, mu):
        """Returns the constants.
        """
        with tf.variable_scope('constant'):
            _mu = tf.constant(mu, shape=[], dtype=tf.float32)

        return _mu

    def _create_user_terms(self, users):
        """Returns the tensors related to users.
        """
        num_users = self.num_users
        num_factors = self.num_factors

        with tf.variable_scope('user'):
            user_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_users, num_factors],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_p_u))

            user_bias = tf.get_variable(
                name='bias',
                shape=[num_users, ],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_u))

            p_u = tf.nn.embedding_lookup(
                user_embeddings,
                users,
                name='p_u')

            b_u = tf.nn.embedding_lookup(
                user_bias,
                users,
                name='b_u')

        return p_u, b_u

    def _create_item_terms(self, items):
        """Returns the tensors related to items.
        """
        num_items = self.num_items
        num_factors = self.num_factors

        with tf.variable_scope('item'):
            item_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_q_i))

            item_bias = tf.get_variable(
                name='bias',
                shape=[num_items, ],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(self.reg_b_i))

            q_i = tf.nn.embedding_lookup(
                item_embeddings,
                items,
                name='q_i')

            b_i = tf.nn.embedding_lookup(
                item_bias,
                items,
                name='b_i')

        return q_i, b_i

    def _create_prediction(self, mu, b_u, b_i, p_u, q_i):
        """Returns the tensor of prediction.

           Note that the prediction 
            r_hat = \mu + b_u + b_i + p_u * q_i
        """
        with tf.variable_scope('prediction'):
            pred = tf.reduce_sum(
                tf.multiply(p_u, q_i),
                axis=1)

            pred = tf.add_n([b_u, b_i, pred])

            pred = tf.add(pred, mu, name='pred')

        return pred

    def _create_loss(self, pred, ratings):
        """Returns the L2 loss of the difference between
            ground truths and predictions.

           The formula is here:
            L2 = sum((r - r_hat) ** 2) / 2
        """
        with tf.variable_scope('loss'):
            loss = tf.nn.l2_loss(tf.subtract(ratings, pred), name='loss')

        return loss

    def _create_optimizer(self, loss):
        """Returns the optimizer.

           The objective function is defined as the sum of
            loss and regularizers' losses.
        """
        with tf.variable_scope('optimizer'):
            objective = tf.add(
                loss,
                tf.add_n(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)),
                name='objective')

            try:
                optimizer = tf.contrib.keras.optimizers.Nadam(
                ).minimize(objective, name='optimizer')
            except:
                optimizer = tf.train.AdamOptimizer().minimize(objective, name='optimizer')

        return optimizer

    def _build_graph(self, mu):
        _mu = self._create_constants(mu)

        self._users, self._items, self._ratings = self._create_placeholders()

        p_u, b_u = self._create_user_terms(self._users)
        q_i, b_i = self._create_item_terms(self._items)

        self._pred = self._create_prediction(_mu, b_u, b_i, p_u, q_i)

        loss = self._create_loss(self._ratings, self._pred)

        self._optimizer = self._create_optimizer(loss)

        self._built = True

    def _run_train(self, x, y, epochs, batch_size, validation_data):
        train_gen = BatchGenerator(x, y, batch_size)
        steps_per_epoch = np.ceil(train_gen.length / batch_size).astype(int)

        self._sess.run(tf.global_variables_initializer())

        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))

            pbar = utils.Progbar(steps_per_epoch)

            for step, batch in enumerate(train_gen.next(), 1):
                users = batch[0][:, 0]
                items = batch[0][:, 1]
                ratings = batch[1]

                self._sess.run(
                    self._optimizer,
                    feed_dict={
                        self._users: users,
                        self._items: items,
                        self._ratings: ratings
                    })

                pred = self.predict(batch[0])

                update_values = [
                    ('rmse', rmse(ratings, pred)),
                    ('mae', mae(ratings, pred))
                ]

                if validation_data is not None and step == steps_per_epoch:
                    valid_x, valid_y = validation_data
                    valid_pred = self.predict(valid_x)

                    update_values += [
                        ('val_rmse', rmse(valid_y, valid_pred)),
                        ('val_mae', mae(valid_y, valid_pred))
                    ]

                pbar.update(step, values=update_values,
                            force=(step == steps_per_epoch))

    def train(self, x, y, epochs=100, batch_size=1024, validation_data=None):

        if x.shape[0] != y.shape[0] or x.shape[1] != 2:
            raise ValueError('The shape of x should be (samples, 2) and '
                             'the shape of y should be (samples, 1).')

        if not self._built:
            self._build_graph(np.mean(y))

        self._run_train(x, y, epochs, batch_size, validation_data)

    def predict(self, x):
        if not self._built:
            raise RunTimeError('The model must be trained '
                               'before prediction.')

        if x.shape[1] != 2:
            raise ValueError('The shape of x should be '
                             '(samples, 2)')

        pred = self._sess.run(
            self._pred,
            feed_dict={
                self._users: x[:, 0],
                self._items: x[:, 1]
            })

        pred = pred.clip(min=self.min_value, max=self.max_value)

        return pred
