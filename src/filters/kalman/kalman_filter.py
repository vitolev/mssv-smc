import numpy as np
from src.models.lgm import LGModel, LGModelParams

class KalmanFilter:
    """
    Kalman Filter for LGModel.
    """
    def __init__(self):
        pass

    def filter(self, y, theta: LGModelParams):
        """
        Run the Kalman filter on observation sequence y to obtain filtering distributions p(x_t | y_{1:t}).

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : LGModelParams
            Model parameters.

        Returns
        -------
        mu : array-like, shape (T,)
            Filtering means over time.
        var : array-like, shape (T,)
            Filtering variances over time.
        """
        y = y.flatten()
        T = len(y)
        mu = np.zeros(T)
        var = np.zeros(T)

        # Initial state
        mu_prev = 0.0
        var_prev = 100.0

        for t in range(T):
            # Prediction
            mu_pred = theta.a * mu_prev
            var_pred = theta.a**2 * var_prev + theta.sigma_x**2

            # Update
            K = var_pred * theta.b / (theta.b**2 * var_pred + theta.sigma_y**2)
            mu[t] = mu_pred + K * (y[t] - theta.b * mu_pred)
            var[t] = (1 - K * theta.b) * var_pred

            mu_prev = mu[t]
            var_prev = var[t]

        return mu, var
    
    def log_marginal_likelihood(self, y, theta: LGModelParams):
        """
        Compute the log marginal likelihood log p(y | theta) using the Kalman filter.

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : LGModelParams
            Model parameters.

        Returns
        -------
        logmarlik : float
            Log marginal likelihood of the observations given the parameters.
        """
        y = y.flatten()
        T = len(y)

        # Initial state prior
        mu_pred = 0.0
        var_pred = 100.0

        loglik = 0.0

        for t in range(T):

            # Observation prediction based on x_t
            y_mean = theta.b * mu_pred
            S = theta.b**2 * var_pred + theta.sigma_y**2

            loglik += -0.5 * (
                np.log(2 * np.pi)
                + np.log(S)
                + (y[t] - y_mean)**2 / S
            )

            # Kalman update
            K = var_pred * theta.b / S
            mu_filt = mu_pred + K * (y[t] - y_mean)
            var_filt = (1 - K * theta.b) * var_pred

            # Predict next state
            mu_pred = theta.a * mu_filt
            var_pred = theta.a**2 * var_filt + theta.sigma_x**2

        return loglik


    def smoother(self, y, theta: LGModelParams):
        """
        Run the Kalman smoother on observation sequence y to obtain smoothing distributions p(x_t | y_{1:T}).

        Parameters
        ----------
        y : array-like, shape (T,)
            Observations over time.
        theta : LGModelParams
            Model parameters.

        Returns
        -------
        mu_smooth : array-like, shape (T,)
            Smoothing means over time.
        var_smooth : array-like, shape (T,)
            Smoothing variances over time.
        """
        # First run the filter to get the necessary quantities for smoothing
        mu_filter, var_filter = self.filter(y, theta)

        T = len(y)
        mu_smooth = np.zeros(T)
        var_smooth = np.zeros(T)

        # Initialize with the last filtering distribution
        mu_smooth[-1] = mu_filter[-1]
        var_smooth[-1] = var_filter[-1]

        for t in range(T-2, -1, -1):
            # Compute the smoothing gain
            C = var_filter[t] * theta.a / (theta.a**2 * var_filter[t] + theta.sigma_x**2)

            # Update the smoothed estimates
            mu_smooth[t] = mu_filter[t] + C * (mu_smooth[t+1] - theta.a * mu_filter[t])
            var_smooth[t] = var_filter[t] + C**2 * (var_smooth[t+1] - theta.a**2 * var_filter[t] - theta.sigma_x**2)

        return mu_smooth, var_smooth
