from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from asymmetric_kde.asymmetric_kde import ImproperGammaEstimator, ProperGammaEstimator
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('--seed', type=int, default=123)
ap.add_argument('plots', nargs='*')
args = ap.parse_args()

# Determine which plots to do
plots = args.plots or ['improper-gamma', 'proper-gamma', 'bandwidth-comparison']

# Set up the plotting parameters
width_pt = 360
width = width_pt / 72
aspect = 3./4
height = width * aspect

rcParams['font.size'] = 9
rcParams['legend.fontsize'] = 'medium'

# Generate samples
np.random.seed(args.seed)
log_mean = 1
log_std = 1
distribution = stats.lognorm(log_std, scale=np.exp(log_mean))
samples = distribution.rvs(size=300)

if 'improper-gamma' in plots:
    # Fit a density estimator
    kde = ImproperGammaEstimator(samples, 'plugin')

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)

    # Plot the original distribution and KDEs
    x = np.linspace(1e-4, 15, 500)
    ax.plot(x, kde(x), color='k', label='Improper gamma estimator')
    ax.plot(x, distribution.pdf(x), color='k', ls=':', label='Generating distribution')

    ax.scatter(samples, np.zeros_like(samples), marker='|', color='k')

    # Finally plot the approximation with a Gaussian
    kde_sample_smoothing = kde.to_variable_gaussian()
    ax.plot(x, kde_sample_smoothing.evaluate(x), color='k', ls='--', label='Gaussian approximation')

    ax.set_xlim(-1,15)
    ax.set_xlabel('Random variable $X$')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig('../paper/improper-gamma.pdf', bbox_inches='tight')
    fig.savefig('../paper/improper-gamma.ps', bbox_inches='tight')
    fig.show()

if 'proper-gamma' in plots:
    # Fit a density estimator
    kde = ProperGammaEstimator(samples, 'plugin')

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)

    # Plot the original distribution and KDEs
    x = np.linspace(1e-4, 15, 500)
    ax.plot(x, kde(x), color='k', label='Proper gamma estimator')
    ax.plot(x, distribution.pdf(x), color='k', ls=':', label='Generating distribution')

    ax.scatter(samples, np.zeros_like(samples), marker='|', color='k')

    # Finally plot the approximation with a Gaussian
    kde_sample_smoothing = kde.to_variable_gaussian()
    ax.plot(x, kde_sample_smoothing.evaluate(x), color='k', ls='--', label='Gaussian approximation')

    ax.set_xlim(-1,15)
    ax.set_xlabel('Random variable $X$')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    fig.savefig('../paper/proper-gamma.pdf', bbox_inches='tight')
    fig.savefig('../paper/proper-gamma.ps', bbox_inches='tight')
    fig.show()

raw_input("[Press ENTER to quit.]")

