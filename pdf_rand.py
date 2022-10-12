# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Lorenzo Zaffina (l.zaffina@studenti.unipi.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


'''
Module: basic Python
Assignment #4 (October 7, 2021)


--- Goal
Create a ProbabilityDensityFunction class that is capable of throwing
preudo-random number with an arbitrary distribution.

(In practice, start with something easy, like a triangular distribution---the
initial debug will be easier if you know exactly what to expect.)


--- Specifications
- the signature of the constructor should be __init__(self, x, y), where
  x and y are two numpy arrays sampling the pdf on a grid of values, that
  you will use to build a spline
- [optional] add more arguments to the constructor to control the creation
  of the spline (e.g., its order)
- the class should be able to evaluate itself on a generic point or array of
  points
- the class should be able to calculate the probability for the random
  variable to be included in a generic interval
- the class should be able to throw random numbers according to the distribution
  that it represents
- [optional] how many random numbers do you have to throw to hit the
  numerical inaccuracy of your generator?
  '''


from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """Class describing a probability density function.
    Parameters
    ----------
    x : array-like
        The array of x values to be passed to the pdf.
    y : array-like
        The array of y values to be passed to the pdf.
    """

    def __init__(self, x, y):
        """Constructor
        """
        super().__init__(x, y)


if __name__ == '__main__':
    x = np.linspace(0., np.pi, 20)
    y = np.sin(x)
    f = ProbabilityDensityFunction(x, y)

    print(f.integral(0., np.pi)) #vorrei che sia normalizzata

    plt.plot(x, y, 'o')
    _x = np.linspace(0., np.pi, 200)
    plt.plot(_x, f(_x), label='spline')
    plt.plot(_x, np.sin(_x), label='sin')
    plt.legend()
    plt.show()
