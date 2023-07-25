import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    best_tau = None
    best_mse = float('inf')

    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau)
        lwr.fit(x_train, y_train)
        y_pred_val, theta = lwr.predict(x_val)
        mse_val = np.mean((y_pred_val - y_val) ** 2)
        print(f'MSE on validation set with tau={tau}: {mse_val}')
        if mse_val < best_mse:
            best_mse = mse_val
            best_tau = tau
        plt.figure()
        plt.plot(x_train[:, 1], y_train, 'bx')
        plt.plot(x_val[:, 1], y_pred_val, 'ro')
        plt.title(f'tau = {tau}, MSE on validation set: {mse_val}')
        plt.show()

    print(f'Best tau value: {best_tau}, with lowest MSE on validation set: {best_mse}')

    lwr_best = LocallyWeightedLinearRegression(best_tau)
    lwr_best.fit(x_train, y_train)
    y_pred_test = lwr_best.predict(x_test)
    MSE_test = np.mean((y_pred_test - y_test) ** 2)
    print(f'MSE on test set with best tau: {best_tau}: {MSE_test}')
    np.savetxt(pred_path, y_pred_test)
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
