#!/usr/bin/python

import numpy as np
import pytest

from itertools import count

NUM_RUNS = 1000
MONTE_CARLO_NUM_POINTS = 400


def main():
    for N in (10, 100):  # N is the number of sample points
        results = [run_experiment(N) for _ in range(NUM_RUNS)]
        num_iterations, p_f_neq_g = np.mean(results, axis=0)
        print('N={}: iter={}, prob={}'.format(N, num_iterations, p_f_neq_g))


def run_experiment(N):
    """
    Runs an experiment with `N` random points.
    """
    f = random_line()
    sample_points = random_points(N)
    correct_labels = label_points(sample_points, f)
    g, num_iterations = perceptron_learning(sample_points, correct_labels)
    p_f_neq_g = calculate_p_f_neq_g(f, g)
    return num_iterations, p_f_neq_g


def calculate_p_f_neq_g(f, g, points=None):
    """
    Calculate the probability P(f(x) != g(x)) using Monte Carlo method.
    `f` and `g` are 3-element arrays [w0, w1, w2] representing a line:
    w0 + w1 * x + w2 * y = 0.
    `points` are used when doing Monte Carlo; if not specified, they are
    generated randomly.
    """
    if points is None:
        points = random_points(MONTE_CARLO_NUM_POINTS)
    label_f = label_points(points, f)
    label_g = label_points(points, g)
    return np.mean(~np.isclose(label_f, label_g))


def perceptron_learning(points, correct_labels):
    w = np.zeros(3)
    for num_iterations in count(start=0, step=1):
        pred_labels = label_points(points, w)
        miss_condition = ~np.isclose(pred_labels, correct_labels)

        # If all points are correctly classified
        if not miss_condition.any():
            break

        miss_point_indexes = miss_condition.nonzero()[0]
        miss_point_index = miss_point_indexes[0] #np.random.choice(miss_point_indexes, 1)[0]
        miss_point = points[miss_point_index]
        miss_point_correct_label = correct_labels[miss_point_index]
        x = np.hstack((1, miss_point))
        w += miss_point_correct_label * x

    return w, num_iterations


def label_points(points, line):
    """
    Labels a list of points using a line. A label can be either -1, 0, or 1.
    Returns a list of labels corresponding to the points.
    A list of points is represented by a 2-dimensional array `points`.
    A line: w0 + w1 * x + w2 * y = 0 is represented by
    an array `line` [w0, w1, w2].
    """
    num_points = points.shape[0]
    a = np.column_stack((np.ones(num_points), points))
    # sign( [1., x, y] . [w0, w1, w2] )
    return np.sign(a.dot(line))


def find_line(point_1, point_2):
    """
    Finds the line crossing `point_1` and `point_2`. Returns an array
    [w0, w1, w2] representing a line: w0 + w1 * x + w2 * y = 0.
    `point_1` and `point_2` are 2-element arrays.
    """
    # [x1 y1] . [w1] = [-w0] ==> a . x = b
    # [x2 y2]   [w2]   [-w0]
    a = [point_1, point_2]
    w0 = 1.
    b = [-w0, -w0]
    w1, w2 = np.linalg.solve(a, b)
    return np.array([w0, w1, w2])


def random_line():
    point_1, point_2 = random_points(2)
    return find_line(point_1, point_2)
    #return find_line(np.array([-0.40877, -0.653459]), np.array([-0.579892,  -0.84767]))

def random_points(N):
    return np.random.uniform(-1., 1., size=(N, 2))
    #return np.array([[0.308796, -0.758407],[-0.173793, 0.120863],[0.841236, -0.780489],
     #       [-0.291265, 0.129208],[-0.812164, -0.697282]])

# ========================================================================== #

def test_calculate_p_f_neq_g():
    f = np.array([0., 0., 1.])  # y = 0
    g = np.array([-2., 0., 1.])  # y = 2
    points = np.array([[0., -1.], [0., 1.], [0., 3.]])
    result = calculate_p_f_neq_g(f, g, points)
    assert np.isclose(result, 1./3)


def test_perceptron_training():
    points = np.array([[0., 4.1], [1., 1.9], [2., .1]])
    labels = np.array([-1, 1., -1.])
    w, num_iterations = perceptron_learning(points, labels)
    w /= w[0]  # Normalize w so that w0 = 1
    expected = np.array([1., -.5, -.25])
    assert np.allclose(w, expected, atol=.01)


def test_label_points():
    line = np.array([1., -.5, -.25])
    points = np.array([[0., 4.], [2., 0.], [1., 2.], [1., 1.9], [1., 2.1]])
    result = label_points(points, line)
    expected = np.array([0., 0., 0., 1., -1.])
    assert np.allclose(result, expected)


def test_find_line():
    point_1, point_2 = np.array([[0., 4.], [2., 0.]])
    result = find_line(point_1, point_2)
    expected = np.array([1., -.5, -.25])
    assert np.allclose(result, expected)

# ========================================================================== #


if __name__ == '__main__':
    #pytest.main(['-s', __file__])
    main()