# asymmetric_kde

`asymmetric_kde` is a python package built to facilitate kernel density estimation using asymmetric kernels. Asymmetric kernels are particularly useful for density estimation when the domain of the PDF you are trying to estimate is bounded. For example, gross income is non-negative and standard KDEs estimate the income distribution poorly.

A comprehensive summary of asymmetric kernel density estimators and their properties can be found in our publication ["Unified treatment of the asymptotics of asymmetric kernel density estimators"](http://arxiv.org/abs/1512.03188).

## Example

The asymmetric kernel density estimators behave similarly to [`gaussian_kde`](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) in the [`scipy`](http://www.scipy.org/) package. A brief example is shown below and a more comprehensive example can be found in `example.py`.

```python
# Generate 500 samples from a log-normal distribution
samples = np.exp(np.random.normal(1, 1, 500))
# Create an estimator using plugin bandwidth estimation
ige = ImproperGammaEstimator(samples, 'plugin')
# Plot the result
x = np.linspace(0, 15)
plt.plot(x, ige(x))
```

![illustration of asymmetric KDEs]()
