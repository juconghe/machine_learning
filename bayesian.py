from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from Data import Data

"""
CS383: Hw6
Instructor: Ian Gemp
TAs: Scott Jordan, Yash Chandak
University of Massachusetts, Amherst

README:

Feel free to make use of the function/libraries imported
You are NOT allowed to import anything else.

Following is a skeleton code which follows a Scikit style API.
Make necessary changes, where required, to get it correctly running.

Note: Running this empty template code might throw some error because
currently some return values are not as per the required API. You need to
change them.

Good Luck!
"""


class Posterior:
    def __init__(self, limes, cherries, a=2, b=2):
        self.a = a
        self.b = b
        self.limes = limes  # shape: (N,)
        self.cherries = cherries  # scalar int
        self.N = np.shape(self.limes)[0]

    def get_MAP(self):
        """
        compute MAP estimate
        :return: MAP estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        # (c+a−1)/(c+l+a+b−2)
        result = []
        for lime in self.limes:
            h_map = 1 - (self.cherries + self.a - 1)/(self.cherries+lime+self.a+self.b-2)
            result.append(h_map)
        return result

    def get_finite(self):
        """
        compute posterior with finite hypotheses
        :return: estimates for diff. values of lime; shape:(N,)
        """
        # WRITE the required CODE HERE and return the computed values
        new_lime = np.zeros(self.N)
        prior = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        hypotheses = {1: (1.0, 0), 2: (0.75, 0.25), 3: (0.5, 0.5), 4: (0.25, 0.75), 5: (0, 1.0)}
        lime = 1
        for d in self.limes:
            temp_sum = 0.0
            alpha = self.compute_alpha(d, hypotheses, prior)
            for i in range(1, len(hypotheses) + 1):
                p_lime = hypotheses[i][lime] * prior[i - 1]
                product = (hypotheses[i][lime]) ** (d)
                temp_sum += (p_lime * product)
            new_lime[d] = temp_sum * alpha
            # print(new_lime[d])
        return new_lime

    def get_infinite(self):
        """
        compute posterior with beta prior
        :return: estimates for diff. values of lime; shape:(N,)
        """
        result = []
        for N in range(self.N):
            alpha = self.compute_infinite_beta(self.a + N, self.b)
            p_lime = self.compute_infinite_beta(self.a + 1 + N, self.b) / alpha
            result.append(p_lime)

        return result

    def h_given_lime(self, d, h, p, alpha):
        lime = 1
        p_h_given_lime = alpha * (h[lime] ** d) * p
        return p_h_given_lime

    def compute_alpha(self, d, hypotheses, prior):
        alpha = 0.0
        lime = 1
        for i in range(1, len(hypotheses) + 1):
            p_lime = (hypotheses[i][lime] ** d) * prior[i - 1]
            alpha += p_lime
        return 1.0 / alpha

    def compute_infinite_beta(self, a, b):
        beta = gamma(a) * gamma(b) / gamma(a+b)
        return beta

if __name__ == '__main__':
    # Get data
    data = Data()
    limes, cherries = data.get_bayesian_data()

    # Create class instance
    posterior = Posterior(limes=limes, cherries=cherries)

    # PLot the results
    plt.plot(limes, posterior.get_MAP(), label='MAP')
    plt.plot(limes, posterior.get_finite(), label='5 Hypotheses')
    plt.plot(limes, posterior.get_infinite(), label='Bayesian with Beta Prior')
    plt.legend()
    plt.savefig('figures/Q4.png')
