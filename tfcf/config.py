import numpy as np


class Config(object):
    """Configuration class for collaborative filtering model
    """

    num_users = None
    num_items = None
    num_factors = 15

    # minimum and maximum value of prediction for clipping
    min_value = -np.inf
    max_value = np.inf

    # regularization scale
    reg_b_u = 0.0001
    reg_b_i = 0.0001
    reg_p_u = 0.005
    reg_q_i = 0.005
    reg_y_u = 0.005
    reg_g_i = 0.005
