import numpy as np
import numbers
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


class KernelDensityEstimator(object):
    """
    Base class for kernel density estimators.

    Parameters
    ----------
    samples : list
        array of samples to estimate from
    bandwidth_method : str, scalar or callable
        method used to calculate the bandwidth

    Attributes
    ----------
    samples : list
        array of samples to estimate from
    bandwidth_method : str, scalar or callable
        method used to calculate the bandwidth
    n : integer
        number of samples
    bandwidth : scalar
        estimated bandwidth of the kernel
    """

    def __init__(self, samples, bandwidth_method=None):
        # Store all the values
        self.samples = np.asarray(samples)
        self.n = len(self.samples)
        self.bandwidth_method = bandwidth_method
        self.bandwidth = None

        # Pick a bandwidth
        if self.bandwidth_method == 'cross_validation':
            self.bandwidth = self.minimize_cv_score()
        elif self.bandwidth_method == 'plugin':
            self.bandwidth = self.plugin()
        elif isinstance(self.bandwidth_method, numbers.Number):
            self.bandwidth = self.bandwidth_method
            self.bandwidth_method = 'numeric'
        elif callable(self.bandwidth_method):
            self.bandwidth = self.bandwidth_method(self.samples)

    def evaluate(self, points, bandwidth=None):
        """
        Estimate the PDF at a set of points.

        Parameters
        ----------
        points : list
            array of points at which to estimate the PDF
        bandwidth : scalar, optional
            smoothing bandwidth of the kernel

        Returns
        -------
        values : (# of points)-array
            values of the estimated PDF at each point
        """
        # Evaluate the density at all points
        contributions = self.evaluate_kernel(bandwidth or self.bandwidth,
                                             self.samples, points)
        # Sum up the contributions
        return np.mean(contributions, axis=1)

    __call__ = evaluate

    def evaluate_kernel(self, bandwidth, samples, points):
        """
        Evaluate the kernel at a set of points.

        Parameters
        ----------
        bandwidth : scalar
            smoothing bandwidth of the kernel
        samples : list
            array of samples to estimate from
        points : list
            array of points at which to evaluate the kernel

        Returns
        -------
        values : (# of points, # of samples)-array
            values of kernels for each sample and point combination
        """
        raise NotImplementedError

    def plugin(self):
        """
        Evaluate the bandwidth using a plugin method.

        Returns
        -------
        bandwidth : scalar
            optimal smoothing bandwidth
        """
        raise NotImplementedError

    def evaluate_kernel_product_integral(self, bandwidth, samples1, samples2):
        """
        Evaluate the integral of the product of two kernels.

        Parameters
        ----------
        bandwidth : scalar
            smoothing bandwidth of the kernel
        samples1 : list
            array of samples to estimate from
        samples2 : list
            array of samples to estimate from

        Returns
        -------
        values : (# of samples, # of samples)-array
            values of the integral
        """
        raise NotImplementedError

    def evaluate_cv_score(self, bandwidth):
        """
        Evaluate the cross-validation score.

        Parameters
        ----------
        bandwidth : scalar
            smoothing bandwidth of the kernel

        Returns
        -------
        score : scalar
            cross-validation score
        """
        # Return a high penalty for non-positive bandwidths
        if bandwidth <= 0:
            return np.inf
        try:
            # Evaluate the integral of the product of the kernels for all points
            contributions = self.evaluate_kernel_product_integral(bandwidth, self.samples, self.samples)
            score = np.mean(contributions)
        except NotImplementedError:
            raise NotImplementedError('`evaluate_kernel_product_integral` must be implemented for cross validation.')
        # Evaluate the estimate at all data points
        contributions = self.evaluate_kernel(bandwidth, self.samples, self.samples)
        contributions[range(self.n), range(self.n)] = 0
        # Obtain the part due to the linear terms
        score -= 2 * np.sum(contributions) / (self.n * (self.n - 1))
        return score

    def minimize_cv_score(self):
        """
        Minimize the cross-validation score.

        Returns
        -------
        bandwidth : scalar
            optimal smoothing bandwidth
        """
        return minimize_scalar(self.evaluate_cv_score)['x']


class GaussianEstimator(KernelDensityEstimator):
    """
    Standard Gaussian kernel density estimator.
    """
    def __init__(self, samples, bandwidth_method='plugin'):
        super(GaussianEstimator, self).__init__(samples, bandwidth_method)

    def plugin(self):
        return 1.06 * np.std(self.samples) * self.n ** -0.2

    def evaluate_kernel(self, bandwidth, samples, points):
        # Evaluate the exponent of the gaussian kernel
        chi2 = (samples[None, :] - points[:, None]) ** 2
        # Apply the exponential function and normalise
        return np.exp(-.5 * chi2 / bandwidth**2) \
            / (np.sqrt(2 * np.pi) * bandwidth)

    def evaluate_kernel_product_integral(self, bandwidth, samples1, samples2):
        # The integral can be performed analytically for Gaussians
        return self.evaluate_kernel(np.sqrt(2) * bandwidth, samples1, samples2)


class VariableGaussianEstimator(KernelDensityEstimator):
    """
    Gaussian kernel density estimator with shift function and variable bandwidth.

    Parameters
    ----------
    samples : list
        array of samples to estimate from
    bandwidth_function : callable
        function to evaluate the variable bandwidth
    shift_function : callable
        function to evaluate the variable shift
    bandwidth_method : str, scalar or callable
        method used to calculate the bandwidth

    Attributes
    ----------
    samples : list
        array of samples to estimate from
    bandwidth_function : callable
        function to evaluate the variable bandwidth
    shift_function : callable
        function to evaluate the variable shift
    bandwidth_method : str, scalar or callable
        method used to calculate the bandwidth
    n : integer
        number of samples
    bandwidth : scalar
        estimated bandwidth of the kernel
    """
    def __init__(self, samples, bandwidth_function, shift_function, bandwidth_method=None):
        super(VariableGaussianEstimator, self).__init__(samples, bandwidth_method)
        self.bandwidth_function = bandwidth_function
        self.shift_function = shift_function

    def evaluate_kernel(self, bandwidth, samples, points):
        """
        Evaluate the kernel at a set of points.

        Parameters
        ----------
        bandwidth : scalar
            smoothing bandwidth of the kernel
        samples : list
            array of samples to estimate from
        points : list
            array of points at which to evaluate the kernel

        Returns
        -------
        values : (# of points, # of samples)-array
            values of kernels for each sample and point combination

        Notes
        -----
        The shift function is parametrised as in Eq. (9) of arxiv: 1512.03188 and a minus sign
        should be inserted to get the parametrisation as in Eq. (14).
        """
        # Evaluate the shift and variable bandwidth
        shift = self.shift_function(bandwidth, samples, points)
        bandwidth = self.bandwidth_function(bandwidth, samples, points)
        # Evaluate the chi-square
        chi2 = (samples[None, :] - points[:, None] - bandwidth ** 2 * shift) ** 2
        return np.exp(-.5 * chi2 / bandwidth**2) \
            / (np.sqrt(2 * np.pi) * bandwidth)


class ImproperGammaEstimator(KernelDensityEstimator):
    def to_variable_gaussian(self):
        # Convert to a variable Gaussian estimator
        shift_function = lambda bandwidth, _, _2: 1.0
        bandwidth_function = lambda bandwidth, _, points: bandwidth * np.sqrt(bandwidth * bandwidth + points[:, None])
        variable = VariableGaussianEstimator(self.samples, bandwidth_function, shift_function, self.bandwidth)
        return variable

    def evaluate_kernel(self, bandwidth, samples, points):
        # Calculate the shape and scale parameters
        scale = bandwidth ** 2
        if isinstance(points, np.ndarray):
            shape = 1 + points[:, None] / scale
        else:
            shape = 1 + points / scale
        # Evaluate the gamma distribution
        if isinstance(samples, np.ndarray):
            samples = samples[None, :]
        loggamma = (shape-1) * np.log(samples) - samples / scale - shape * np.log(scale) - gammaln(shape)
        return np.exp(loggamma)

    def plugin(self):
        # Compute the logarithmic mean and variance
        log_samples = np.log(self.samples)
        log_mean = np.mean(log_samples)
        log_std = np.std(log_samples)
        # Compute the bandwidth according to the first row, fourth column in  Tbl. 2 in arxiv:1512.03188
        return (2 ** 0.8 * np.exp(log_mean / 2.) * log_std) / (np.exp((17 * log_std ** 2) / 8.) * self.n * \
                                                               (12 + 20 * log_std ** 2 + 9 * log_std ** 4)) ** 0.2

    def evaluate_asymptotic_score(self, bandwidth):
        """
        Evaluate the asymptotic mean integrated squared error.

        Parameters
        ----------
        bandwidth : scalar
            smoothing bandwidth of the kernel
        Returns
        -------
        score : scalar
            asymptotic MISE
        """
        log_samples = np.log(self.samples)
        log_mean = np.mean(log_samples)
        log_std = np.std(log_samples)

        return np.exp(-3 * log_mean + log_std**2/8.) * \
               ((64 * np.exp((5 * log_mean) / 2.)) / self.n + (np.exp((17 * log_std ** 2) / 8.) * bandwidth ** 5 *
               (12 + 20 * log_std ** 2 + 9 * log_std ** 4)) / log_std ** 5) / (128. * np.sqrt(np.pi) * bandwidth)


class ProperGammaEstimator(KernelDensityEstimator):
    def to_variable_gaussian(self):
        # The shift function has a minus sign because the VariableGaussianEstimator
        # is parametrised for improper estimators
        shift_function = lambda bandwidth, _, _2: -1.0
        bandwidth_function = lambda bandwidth, samples, _: bandwidth * np.sqrt(bandwidth * bandwidth + samples[None, :])
        variable = VariableGaussianEstimator(self.samples, bandwidth_function, shift_function, self.bandwidth)
        return variable

    def evaluate_kernel(self, bandwidth, samples, points):
        # Calculate the shape and scale parameters
        scale = bandwidth ** 2
        if isinstance(samples, np.ndarray):
            shape = 1 + samples[None] / scale
        else:
            shape = 1 + samples / scale
        # Evaluate the gamma distribution
        if isinstance(points, np.ndarray):
            points = points[:, None]
        loggamma = (shape-1) * np.log(points) - points / scale - shape * np.log(scale) - gammaln(shape)
        return np.exp(loggamma)

    def evaluate_kernel_product_integral(self, bandwidth, samples1, samples2):
        # Calculate the shape and scale parameters
        scale = bandwidth ** 2
        shape1 = 1 + samples1[None, :] / scale
        shape2 = 1 + samples2[:, None] / scale
        # Evaluate
        loggamma = (1 - shape1 - shape2) * np.log(2) + gammaln(shape1 + shape2 - 1) \
            - gammaln(shape1) - gammaln(shape2)
        return np.exp(loggamma) / scale

    def plugin(self):
        # Compute the logarithmic mean and variance
        log_samples = np.log(self.samples)
        log_mean = np.mean(log_samples)
        log_std = np.std(log_samples)

        return (2 ** 0.8 * np.exp(log_mean / 2.) * log_std) / (np.exp((17 * log_std ** 2) / 8.) *
                                                               self.n * (12 + 4 * log_std ** 2 + log_std ** 4)) ** 0.2

    def evaluate_asymptotic_score(self, bandwidth):
        """
        Evaluate the asymptotic mean integrated squared error.

        Parameters
        ----------
        bandwidth : scalar
            smoothing bandwidth of the kernel
        Returns
        -------
        score : scalar
            asymptotic MISE
        """
        log_samples = np.log(self.samples)
        log_mean = np.mean(log_samples)
        log_std = np.std(log_samples)

        return (np.exp(-3 * log_mean + log_std ** 2 / 8.) * ((64 * np.exp((5 * log_mean) / 2.)) / self.n +
               (np.exp((17 * log_std ** 2) / 8.) * bandwidth ** 5 * (12 + 4 * log_std ** 2 + log_std ** 4)) /
               log_std ** 5)) / (128. * np.sqrt(np.pi) * bandwidth)
