# Copyright (C) 2022 Luca Baldini (luca.baldini@pi.infn.it)
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

"""Unit test for the pdf.
"""

import unittest
import sys


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
if sys.flags.interactive:
    plt.ion()



from  pdf_rand.pdf import ProbabilityDensityFunction

class TestPdf(unittest.TestCase):

    """
    """

    def test_uniform(self):
        """
        """
        x = np.linspace(0., 1., 100)
        y = np.full(x.shape, 1.)
        pdf = ProbabilityDensityFunction(x, y)
        self.assertAlmostEqual(pdf(0.5), 1.)
        self.assertAlmostEqual(pdf.integral(0., 1.), 1.)
        self.assertAlmostEqual(pdf.prob(0.25, 0.75), 0.5)

    def test_triangular(self):
        x = np.linspace(0., 1., 100)
        y = 4. * x
        pdf = ProbabilityDensityFunction(x, y)
        plt.figure('Triangular pdf')
        plt.plot(x, pdf(x))
        plt.figure('Triangular cdf')
        plt.plot(x, pdf.cdf(x))
        plt.figure('Triangular ppf')
        plt.plot(x, pdf.ppf(x))
        r = pdf.rnd(1000000)
        plt.figure('Triangular random variate')
        plt.hist(r, bins=200)

    def test_fancy(self):
        """
        """
        x = np.linspace(0., 1., 10000)
        y = np.zeros(x.shape)
        y[x <= 0.5] = 2. * x[x <= 0.5]
        y[x > 0.75] = 3.
        pdf = ProbabilityDensityFunction(x, y, 1)
        plt.figure('Fancy pdf')
        plt.plot(x, pdf(x))
        print(pdf.integral(0., 1.))
        plt.figure('Fancy cdf')
        plt.plot(x, pdf.cdf(x))
        plt.figure('Fancy ppf')
        plt.plot(x, pdf.ppf(x))
        r = pdf.rnd(1000000)
        plt.figure('Fancy random variate')
        plt.hist(r, bins=200)




if __name__ == '__main__':
    unittest.main(exit=not sys.flags.interactive)