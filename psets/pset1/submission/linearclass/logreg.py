import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    util.plot(x_valid, y_valid, clf.theta, 'path_to_save')

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, clf.predict(x_valid))
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Theta initialization as a zero vector
        self.theta = np.zeros(x.shape[1])

        # Auxiliary functions
        def hypothesis(x):
            return 1 / (1 + np.exp(-np.dot(x, self.theta)))
        
        def loss(x, y):
            h = hypothesis(x)
            return -np.sum(y * np.log(h + 1e-5) + (1 - y) * np.log(1 - h + 1e-5))

        # Theta update using Newton's method
        for i in range(self.max_iter):
            h = hypothesis(x)
            gradient = -np.dot(x.T, y - h)
            Hessian = sum(h[i] * (1 - h[i]) * np.outer(x[i], x[i]) for i in range(x.shape[0]))
            theta_new = self.theta - self.step_size * np.dot(np.linalg.inv(Hessian), gradient)

            if np.linalg.norm(self.theta - theta_new, 1) < self.eps:
                break

            self.theta = theta_new

            if self.verbose and i % 10000 == 0:
                print(f"Iteration: {i}, Loss: {loss(x, y)}")
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-np.dot(x, self.theta))) 
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')