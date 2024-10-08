import numpy as np

# import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import (
    MiniBatchDictionaryLearning as SklearnMiniBatchDictionaryLearning,
)
import multiprocessing as mp
import time
from numpy.linalg import pinv
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import sparse_encode
import warnings, wandb
from functools import partial

warnings.filterwarnings("ignore")
# Load the Olivetti faces dataset
data = fetch_olivetti_faces()
faces = data.images
n_samples, height, width = faces.shape
n_features = height * width
faces = faces.reshape((n_samples, n_features))
faces_centered = faces - faces.mean(axis=0)
# faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)


# # Function to visualize faces
# def plot_faces(images, titles=None, h=64, w=64, n_row=3, n_col=6):
#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
#     plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
#         plt.title(titles[i] if titles is not None else "", size=12)
#         plt.xticks(())
#         plt.yticks(())


# plot_faces(faces)
# plt.savefig("olivetti_faces.png", dpi=300)


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
        init_value=1.0,
        n=2,  # hyper-parameters for u^2n-v^2n parameterization
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.tol = tol
        self.init_value = init_value  # x
        self.n = n

        self.sc_solver_type = SC_solver
        if SC_solver == "lasso":
            self.SC_solver = self.lasso_lars_solver
        elif SC_solver == "lasso_lar":
            self.SC_solver = self.Lasso_solver
        elif SC_solver == "ista":
            self.SC_solver = self.ista_solver
        elif SC_solver == "power_solver":
            self.SC_solver = self.power_solver

        else:
            raise ValueError("SC_solver should be 'lasso' or 'lasso_lar' or 'ista'")

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
                rho = np.dot(dictionary[j], residual)
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

    def soft_threshold(self, X, alpha):
        return np.sign(X) * np.maximum(np.abs(X) - alpha, 0)

    def ista_solver(self, X, max_iter=10, tol=1e-8):

        codes = np.dot(X, self.dictionary_.T)

        lipschitz_const = np.linalg.norm(self.dictionary_, ord=2) ** 2
        step_size = 1.0 / lipschitz_const

        residual = np.dot(codes, self.dictionary_) - X

        for _ in range(max_iter):
            codes_old = codes.copy()
            grad = np.dot(
                residual, self.dictionary_.T
            )  # Corrected gradient calculation
            codes = self.soft_threshold(
                codes - 2 * step_size * grad, self.alpha * step_size
            )
            residual = np.dot(codes, self.dictionary_) - X
            if np.linalg.norm(codes - codes_old) < tol:
                break

        return codes

    def power_solver(
        self,
        X,
        codes_U_batch=None,
        codes_V_batch=None,
        max_iter=10,  # 1
        tol=1e-8,
        return_mw=False,
    ):
        # init code
        if codes_U_batch is not None and codes_V_batch is not None:
            codes_U = codes_U_batch
            # codes_w = np.dot(X, self.dictionary_.T)
            codes_V = codes_V_batch

        else:
            n_samples = X.shape[0]
            if self.init_value == "normal":
                X_init = np.random.normal(0, 1.0, (n_samples, self.n_components))

            else:
                X_init = np.ones((n_samples, self.n_components)) * self.init_value
            # X_init = np.ones((n_samples, self.n_components)) * self.init_value
            codes_U = np.power(
                (X_init + np.sqrt(X_init**2 + 1)) / 2,
                1 / (2 * self.n),
            )
            codes_V = np.power(
                (-X_init + np.sqrt(X_init**2 + 1)) / 2,
                1 / (2 * self.n),
            )

        codes = np.power(codes_U, 2 * self.n) - np.power(codes_V, 2 * self.n)

        residual = np.dot(codes, self.dictionary_) - X
        # lipschitz_const = np.linalg.norm(
        #     np.dot(X, self.dictionary_.T).flatten(), ord=np.inf
        # )
        lipschitz_const = np.linalg.norm(self.dictionary_, ord=2) ** 2
        step_size = 0.001 / lipschitz_const
        # print(f"lipschitz_const: {lipschitz_const},step_size: {step_size}")
        for _ in range(max_iter):
            codes_U_old = codes_U.copy()
            codes_V_old = codes_V.copy()
            codes_old = codes.copy()
            # grad_U = 4n(D^T*residual)*u^{2n-1}+2*lamb*n*u^{2n-1}
            grad_U = 2 * np.dot(residual, self.dictionary_.T) * (
                np.power(codes_U_old, 2 * self.n - 1)
            ) + self.alpha * (np.power(codes_U_old, 2 * self.n - 1))
            codes_U = codes_U_old - step_size * 2 * self.n * grad_U

            # grad_V = -4n(D^T*residual)*v^{2n-1}+2*lamb*n*v^{2n-1}
            grad_V = -2 * np.dot(residual, self.dictionary_.T) * (
                np.power(codes_V_old, 2 * self.n - 1)
            ) + self.alpha * (np.power(codes_V_old, 2 * self.n - 1))
            codes_V = codes_V_old - step_size * 2 * self.n * grad_V

            # cal new code
            codes = np.power(codes_U, 2 * self.n) - np.power(codes_V, 2 * self.n)
            residual = np.dot(codes, self.dictionary_) - X
            # print(
            #     f"mw sparse coding iter {_}, residual norm: {np.linalg.norm(residual)}"
            # )
            if np.linalg.norm(codes - codes_old) < tol:
                break

        if return_mw:
            return codes, codes_U, codes_V
        else:
            return codes

    def fit(self, X):
        n_samples, _ = X.shape
        data_indices = np.arange(n_samples)
        self._initialize_dict(X)

        a_prev = 0.01 * np.identity(self.n_components)
        b_prev = 0
        if self.init_value == "normal":
            X_init = np.random.normal(0, 1.0, (n_samples, self.n_components))
        else:
            X_init = np.ones((n_samples, self.n_components)) * self.init_value
        codes_U_X = np.power(
            (X_init + np.sqrt(X_init**2 + 1**2)) / 2,
            1 / (2 * self.n),
        )
        codes_V_X = np.power(
            (-X_init + np.sqrt(X_init**2 + 1**2)) / 2,
            1 / (2 * self.n),
        )
        print("codes_U_X:\r\n", codes_U_X)
        print("codes_V_X:\r\n", codes_V_X)
        for iteration in range(self.n_iter):
            # log results
            if iteration > 0:
                codes_old = codes.copy()

            codes = np.power(codes_U_X, 2 * self.n) - np.power(codes_V_X, 2 * self.n)
            if iteration > 0:
                wandb.log(
                    {
                        "code_in - code_t frob norm": np.linalg.norm(
                            codes - codes_old, ord="fro"
                        ),
                    },
                    step=iteration + 1,
                )
            custom_reconstruction = np.dot(codes, self.dictionary_)
            custom_mse = mean_squared_error(X, custom_reconstruction)
            code_frob_norm = np.linalg.norm(codes, ord="fro")
            code_nuc_norm = np.linalg.norm(codes, ord="nuc")
            print(
                f"Iteration {iteration+1}, error: {custom_mse:.6f}, code frob norm: {code_frob_norm}, nuc norm: {code_nuc_norm}, other: {np.sum(np.abs(codes))}"
            )
            wandb.log(
                {
                    "reconstruction MSE": custom_mse,
                    "code_frob_norm": code_frob_norm,
                    "code_nuc_norm": code_nuc_norm,
                    "other": np.sum(np.abs(codes)),
                },
                step=iteration + 1,
            )
            # one update
            np.random.shuffle(data_indices)
            for batch_start in range(0, n_samples, self.batch_size):

                batch_indices = data_indices[
                    batch_start : batch_start + self.batch_size
                ]
                X_batch = X[batch_indices]
                codes_U_batch = codes_U_X[batch_indices]
                codes_V_batch = codes_V_X[batch_indices]

                # sparse coding update
                codes_batch, codes_m_X_batch, codes_w_X_batch = self.SC_solver(
                    X_batch, codes_U_batch, codes_V_batch, max_iter=1, return_mw=True
                )
                codes_U_X[batch_indices] = codes_m_X_batch
                codes_V_X[batch_indices] = codes_w_X_batch

                # dictionary update
                a_curr = a_prev + np.einsum("bi,bj->bij", codes_batch, codes_batch)
                b_curr = b_prev + np.einsum("bi,bj->bij", X_batch, codes_batch)
                self._update_dict(A=a_curr, B=b_curr)
                a_prev = a_curr
                b_prev = b_curr

            if custom_mse < self.tol:
                break

    def transform(self, X):
        if self.sc_solver_type == "ista" or self.sc_solver_type == "mw_solver":
            codes = self.SC_solver(X, max_iter=50, tol=1e-8)
        else:
            codes = self.SC_solver(X)
        return codes

    def reconstruct(self, X):
        codes = self.transform(X)
        return np.dot(codes, self.dictionary_)


def main(alpha, n, x_init_value):
    # Parameters
    n_components = 50
    batch_size = 200
    n_iter = 100
    wandb_key = "959f092e8f1f3a1a98bd7e6577c1641bc6bb8e99"
    wandb.login(key=wandb_key)
    wandb.init(
        settings=wandb.Settings(_service_wait=1200),
        project="Continue_Sparse_Coding",
        config={
            "alpha": alpha,
            "n": n,
            "x_init_value": x_init_value,
        },
        name=f"u^2n-v^2n with reg={alpha}, n = {n}, x_init_value = {x_init_value}",
    )
    # # ============ Sklearn built-in Mini-Batch Dictionary Learning, for comparison purpose ============
    # sklearn_dict_learning = SklearnMiniBatchDictionaryLearning(
    #     n_components=n_components,
    #     alpha=alpha,
    #     batch_size=batch_size,
    #     max_iter=n_iter,
    # )
    # sklearn_dict_learning.fit(faces_centered)
    # sklearn_code = sklearn_dict_learning.transform(faces_centered)
    # print(
    #     f'sklearn_code frob norm: {np.linalg.norm(sklearn_code, ord="fro")},nuc norm: {np.linalg.norm(sklearn_code, ord="nuc")}'
    # )
    # sklearn_reconstruction = sklearn_code.dot(
    #     sklearn_dict_learning.components_
    # ) + faces.mean(axis=0)
    # sklearn_mse = mean_squared_error(faces, sklearn_reconstruction)
    # # Print reconstruction errors
    # print("Sklearn Mini-Batch Dictionary Learning MSE:", sklearn_mse)
    # # Visualize the reconstruction
    # plot_faces(sklearn_reconstruction[:18], titles=["Sklearn Reconstruction"] * 18)
    # plt.savefig("sklearn_reconstruction.png", dpi=300)

    # ============ Customised Mini-Batch Dictionary Learning ============
    # "mw_solver","log_solver","UPower_Solver"
    for sc_solver in ["power_solver"]:
        custom_dict_learning = MiniBatchDictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            batch_size=batch_size,
            n_iter=n_iter,
            SC_solver=sc_solver,
            init_value=x_init_value,
            n=n,
        )
        start_time = time.time()
        custom_dict_learning.fit(faces)
        print(
            f"custom implementation with {sc_solver} solver took time: {time.time() - start_time:.2f} seconds"
        )
        wandb.finish()
        custom_reconstruction = custom_dict_learning.reconstruct(faces)
        custom_mse = mean_squared_error(faces, custom_reconstruction)
        # Print reconstruction errors
        print(
            f"Custom Mini-Batch Dictionary Learning with {sc_solver} solver MSE:",
            custom_mse,
        )
    #     # Visualize the reconstruction
    #     plot_faces(custom_reconstruction[:18], titles=["Custom Reconstruction"] * 18)
    #     plt.savefig(f"custom_reconstruction with {sc_solver} solver.png", dpi=300)

    # # close all figures
    # plt.close("all")


if __name__ == "__main__":
    from itertools import product

    alpha_values = [
        0.001
    ]  # [10**x for x in range(-4, 1)]   sparse regularization parameter
    # alpha_values.append(0.0)
    n_orders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    U_init_values = ["normal"]  # 0.1

    #  Create a pool of worker processes
    pool = mp.Pool()

    # Apply the run_main function to each combination of alpha and m_init_value in parallel
    pool.starmap(main, product(alpha_values, n_orders, U_init_values))

    # Close the pool and wait for the tasks to complete
    pool.close()
    pool.join()
