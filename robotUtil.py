import numpy as np
import math
import sys

A = np.asarray([[0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [18.3370000000000, -75.8640000000000, 6.39500000000000, 0, 0, 0],
                [-22.1750000000000, 230.549000000000, -49.0100000000000, 0, 0, 0],
                [4.35300000000000, -175.393000000000, 95.2900000000000, 0, 0, 0]])
B = np.asarray([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0.292000000000000, -0.785000000000000, 0.558000000000000],
                [-0.785000000000000, 2.45700000000000, -2.17800000000000],
                [0.558000000000000, -2.17800000000000, 2.60100000000000]
                ])
K = np.asarray([[929.79, 322.082, 73.883, 317.782, 142.912, 49.865],
                [521.09, 431.75, 74.787, 188.568, 105.95, 38.039],
                [277, 239.326, 202.902, 108.11, 70.849, 42.213]])


def robotDynamics(x, u, step):
    x_dot = np.dot(A, x.T) + np.dot(B, u)
    new_x = x.T + step * x_dot
    return new_x.T


def robotControl(x):
    """
    control task, perfect
    :param x: current system state
    :return: control sig limit by +-250
    """
    new_u = np.dot(-K, x.T)
    new_u[0][0] = max(-250, min(250, new_u[0][0]))
    new_u[1][0] = max(-250, min(250, new_u[1][0]))
    new_u[2][0] = max(-250, min(250, new_u[2][0]))
    return new_u

def robotControl_v1(x):
    u = robotControl(x)
    # add noise or bias
    return u

def is_safe(x):
    """
    check if state is controllable
    :param x:
    :return:
    """
    if abs(x[0][0]) <= 0.09 and abs(x[0][1]) <= 0.09 and abs(x[0][2]) <= 0.09:
        # if x[0][0]<=0.8 and x[0][0]>=-0.8:
        return True
    else:
        return False


def safeState(x0, duration, step, p):
    """
    check if a state is in SSS
    :param x0:  initial satae
    :param duration:
    :param step:
    :param p: period of control
    :return:
    """
    x = np.asarray([x0])
    time_sim = int(duration / step)
    time_count = 0
    u = robotControl(x)
    safe = True
    while time_count < time_sim:
        time_count += 1
        x = robotDynamics(x, u, step)
        if time_count % p == 0:
            u = robotControl(x)
    if not is_safe(x):
        safe = False

    return safe