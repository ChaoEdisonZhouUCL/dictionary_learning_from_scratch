import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import (
    MiniBatchDictionaryLearning as SklearnMiniBatchDictionaryLearning,
)
import multiprocessing as mp
import time
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import sparse_encode
import warnings

warnings.filterwarnings("ignore")
# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
faces = data.images
n_samples, height, width = faces.shape
n_features = height * width
faces = faces.reshape((n_samples, n_features))
faces_centered = faces - faces.mean(axis=0)
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)


# Function to visualize faces
def plot_faces(images, titles=None, h=64, w=64, n_row=3, n_col=6):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i] if titles is not None else "", size=12)
        plt.xticks(())
        plt.yticks(())


plot_faces(faces)
plt.savefig("olivetti_faces.png", dpi=300)


# -------------- customised mini-batch dictionary learning ------------------
class MiniBatchDictionaryLearning:
    """
    This algo is mainly based on
    1. the paper https://www.di.ens.fr/~fbach/mairal_icml09.pdf and,
    2. the github code: https://github.com/MehdiAbbanaBennani/online-dictionary-learning-for-sparse-coding/blob/master/src/odl.py
    """

    def __init__(
        self,
        n_components,
        alpha=1,
        batch_size=10,
        n_iter=1000,
        tol=1e-8,
        SC_solver="lasso",  # "lasso" or "lasso_lar"
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.tol = tol
        if SC_solver == "lasso":
            self.SC_solver = self.lasso_lars_solver
        elif SC_solver == "lasso_lar":
            self.SC_solver = self.Lasso_solver

    def _initialize_dict(self, X):
        epsilon = 1e-5
        n_features = X.shape[1]
        self.dictionary_ = np.random.randn(self.n_components, n_features)
        self.dictionary_ /= (
            epsilon + np.linalg.norm(self.dictionary_, axis=1)[:, np.newaxis]
        )

    def _update_dict(self, A, B):
        epsilon = 1e-5
        for i in range(self.batch_size):
            a = A[i]
            b = B[i]

            for j in range(self.n_components):
                u_j = (b[:, j] - np.matmul(self.dictionary_.T, a[:, j])) / a[
                    j, j
                ] + self.dictionary_.T[:, j]
                self.dictionary_[j, :] = u_j[:, np.newaxis].T / max(
                    [1, epsilon + np.linalg.norm(u_j)]
                )

    def _update_codes(self, X_batch):
        codes = sparse_encode(X_batch, self.dictionary_, alpha=self.alpha)
        return codes

    def lasso_lars_solver(self, X):
        n_samples, _ = X.shape
        n_atoms = self.dictionary_.shape[0]
        alphas = np.zeros((n_samples, n_atoms))
        residuals = X.copy()
        active_sets = [[] for _ in range(n_samples)]

        for _ in range(n_atoms):
            correlations = np.dot(residuals, self.dictionary_.T)
            c = np.max(np.abs(correlations), axis=1)
            mask = c >= self.alpha
            if not np.any(mask):
                break
            j = np.argmax(np.abs(correlations), axis=1)
            for i in range(n_samples):
                if mask[i]:
                    active_sets[i].append(j[i])
                    active_sets[i] = list(set(active_sets[i]))
                    if len(active_sets[i]) > 1:
                        sub_dictionary = self.dictionary_[active_sets[i], :]
                        sub_alphas = (
                            np.linalg.pinv(sub_dictionary @ sub_dictionary.T)
                            @ sub_dictionary
                            @ X[i, :]
                        )
                        alphas[i, active_sets[i]] = sub_alphas
                        residuals[i, :] = X[i, :] - sub_dictionary.T @ sub_alphas
                    else:
                        alphas[i, j[i]] = correlations[i, j[i]] / np.dot(
                            self.dictionary_[j[i], :], self.dictionary_[j[i], :]
                        )
                        residuals[i, :] = (
                            X[i, :] - alphas[i, j[i]] * self.dictionary_[j[i], :]
                        )
        return alphas

    def lasso_coordinate_descent(self, x, dictionary):
        n_atoms = dictionary.shape[0]
        alphas = np.zeros(n_atoms)
        for _ in range(10):
            for j in range(n_atoms):
                # Compute the residual excluding the contribution of the j-th atom
                residual = x - np.dot(dictionary.T, alphas) + alphas[j] * dictionary[j]
                # Update the coefficient for the j-th atom
                rho = alphas[j] + np.dot(dictionary[j], residual)
                # soft-threshold
                if rho < -self.alpha / 2:
                    alphas[j] = (rho + self.alpha / 2) / np.dot(
                        dictionary[j], dictionary[j]
                    )
                elif rho > self.alpha / 2:
                    alphas[j] = (rho - self.alpha / 2) / np.dot(
                        dictionary[j], dictionary[j]
                    )
                else:
                    alphas[j] = 0
        return alphas

    def Lasso_solver(self, X):
        n_samples = X.shape[0]
        codes = np.zeros((n_samples, self.n_components))
        for i in range(n_samples):
            codes[i, :] = self.lasso_coordinate_descent(X[i, :], self.dictionary_)
        return codes

    def fit(self, X):
        n_samples, _ = X.shape
        data_indices = np.arange(n_samples)
        self._initialize_dict(X)

        a_prev = 0.01 * np.identity(self.n_components)
        b_prev = 0
        for iteration in range(self.n_iter):
            np.random.shuffle(data_indices)
            for batch_start in range(0, n_samples, self.batch_size):

                batch_indices = data_indices[
                    batch_start : batch_start + self.batch_size
                ]
                X_batch = X[batch_indices]

                codes = self.SC_solver(X_batch)
                a_curr = a_prev + np.einsum("bi,bj->bij", codes, codes)
                b_curr = b_prev + np.einsum("bi,bj->bij", X_batch, codes)
                self._update_dict(A=a_curr, B=b_curr)
                a_prev = a_curr
                b_prev = b_curr

            custom_reconstruction = self.reconstruct(X)
            custom_mse = mean_squared_error(X, custom_reconstruction)
            print(f"Iteration {iteration+1}, error: {custom_mse:.6f}")
            if custom_mse < self.tol:
                break

    def transform(self, X):
        codes = sparse_encode(X, self.dictionary_, alpha=self.alpha)
        return codes

    def reconstruct(self, X):
        codes = self.transform(X)
        return np.dot(codes, self.dictionary_)


def main():
    # Parameters
    n_components = 50
    alpha = 1e-3
    batch_size = 100
    n_iter = 5

    # ============ Sklearn built-in Mini-Batch Dictionary Learning, for comparison purpose ============
    sklearn_dict_learning = SklearnMiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=n_iter,
    )
    sklearn_dict_learning.fit(faces_centered)
    sklearn_reconstruction = sklearn_dict_learning.transform(faces_centered).dot(
        sklearn_dict_learning.components_
    ) + faces.mean(axis=0)
    sklearn_mse = mean_squared_error(faces, sklearn_reconstruction)
    # Print reconstruction errors
    print("Sklearn Mini-Batch Dictionary Learning MSE:", sklearn_mse)
    # Visualize the reconstruction
    plot_faces(sklearn_reconstruction[:18], titles=["Sklearn Reconstruction"] * 18)
    plt.savefig("sklearn_reconstruction.png", dpi=300)

    # ============ Customised Mini-Batch Dictionary Learning ============
    custom_dict_learning = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        batch_size=batch_size,
        n_iter=n_iter,
        SC_solver="lasso",
    )
    start_time = time.time()
    custom_dict_learning.fit(faces)
    print(f"custom implementation time: {time.time() - start_time:.2f} seconds")
    custom_reconstruction = custom_dict_learning.reconstruct(faces)
    custom_mse = mean_squared_error(faces, custom_reconstruction)
    # Print reconstruction errors
    print("Custom Mini-Batch Dictionary Learning MSE:", custom_mse)
    # Visualize the reconstruction
    plot_faces(custom_reconstruction[:18], titles=["Custom Reconstruction"] * 18)
    plt.savefig("custom_reconstruction.png", dpi=300)

    # close all figures
    plt.close("all")


if __name__ == "__main__":
    main()
