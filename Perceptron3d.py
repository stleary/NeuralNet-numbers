import numpy
import time
import scipy.special
import math


"""
Trying out a 3-input perceptron. Learns after about 50 test points. 
"""

class Perceptron3d:
    """
    In this version the perceptron takes 3 inputs to discover a plane defined by the points
    (2,0,0) (0,4,0) (0,0,6)

    """
    def activate(self, x):
        return 0 if (x + self.bias) < 0.0 else 1

    def __init__(self, inputs=3, learning_rate=.0001, bias=0.0):
        self.bias = bias
        self.learning_rate = learning_rate
        self.wx = numpy.random.random() * 0.99 + .01  # sample from [0.01,1)
        self.wy = numpy.random.random() * 0.99 + .01  # sample from [0.01,1)
        self.wz = numpy.random.random() * 0.99 + .01  # sample from [0.01,1)
        self.activation_function = lambda x: self.activate(x)

    def train(self, x, y, z, target):
        result = self.query(x, y, z)
        if (target != result):
            sum = x + y + z
            self.wx += ((target - result) *  x * self.learning_rate)
            self.wy += ((target - result) * y * self.learning_rate)
            self.wz += ((target - result) * z * self.learning_rate)
        return result

    def query(self, x, y, z):
        out_value = self.activation_function(self.wx * x + self.wy * y + self.wz * z)
        return out_value


def main():
    perceptron = Perceptron3d(bias=0, learning_rate=0.1)
    # to find the equation of a plane, given a couple of points:
    # http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfPlanes.aspx
    # Take 3 points: 2,0,0  0,4,0    0,0,6 can calculate the formula for this plane defined by these points:
    # AC = -2,4,0  BC = 0,-4,6
    #
    # The cross product AC X BC = | i  j  k | = i(24-0) + j(0 - -12) + k(8 - 0) = 24i + 12j +8k = 6i + 3j + 2k
    #                             |-2  4  0 |
    #                             | 0 -4  6 |
    # Solving for the known point A, 6(a1 - 2) + 3(a2 - 0) + 2(a3 - 0) -> 6x - 12 + 3y + 2z = 0 -> 6x + 3y +2z - 12 = 0
    # This is the equation of the plane
    #
    # To find the distance of a random point (x0, y0, z0) to the plane,
    # https://mathinsight.org/distance_point_plane_examples
    # d =  | 6x0 + 3y0 + 2z0 - 12| / sqrt(36 + 9 + 4)
    # If positive, it is on one side of the plane, otherwise it is on the other side.
    #
    i = 0
    while (True):
        i += 1
        x = numpy.random.random() * 99.99 + .01 - 50  # sample from [0.01,10)
        y = numpy.random.random() * 99.99 + .01 - 50 # sample from [0.01,10)
        z = numpy.random.random() * 99.99 + .01 - 50  # sample from [0.01,10)
        distance = (6 * x + 3 * y + 2 * z - 12 ) / 7
        if distance > 0:
            target = 1
        else:
            target = 0
        result = perceptron.train(x, y, z, target)
        if target == result:
            print('{}. {} target: {} result: {} x: {} y: {} z: {} xwt {}  ywt {} zwt {} DISTANCE {}'.
                  format(
                         i,
                         str(result == target),
                         target,
                         result,
                         round(x, 2),
                         round(y, 2),
                         round(z, 2),
                         round(perceptron.wx, 2),
                         round(perceptron.wy, 2),
                         round(perceptron.wz, 2),
                         round(distance)))
        else:
            print('{}. {} target: {} result: {} x: {} y: {} z: {} xwt {}  ywt {} zwt {}                              DISTANCE {}'.
                  format(
                     i,
                     str(result == target),
                     target,
                     result,
                     round(x, 2),
                     round(y, 2),
                     round(z, 2),
                     round(perceptron.wx, 2),
                     round(perceptron.wy, 2),
                     round(perceptron.wz, 2),
                     round(distance)))

        time.sleep(.1)


main()
