import numpy as np
from asymmetric_kde import ImproperGammaEstimator
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde

# Generate 500 samples
distribution = lognorm(1, scale=np.exp(1))
samples = distribution.rvs(500)
# Use an improper gamma estimator with plugin bandwidth estimation
ige = ImproperGammaEstimator(samples, 'plugin')

# Plot the resulting density
x = np.linspace(0, 15, 400)
plt.plot(x, ige(x), label='ImproperGammaEstimator')

# Plot the true density
plt.plot(x, distribution.pdf(x), label='LogNormal(1,1)')

# Plot a naive Gaussian estimate
kde = gaussian_kde(samples)
plt.plot(x, kde(x), label='gaussian_kde')
plt.legend()
plt.savefig('example.png')
plt.show()
