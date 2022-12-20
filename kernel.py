from sklearn.gaussian_process.kernels import RBF
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform
import math

def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and length_scale.shape[0]!=1:
        raise ValueError(
            "Mater Quaternion cannot have more than one lengthscale")
    return length_scale

def _approx_fprime(xk, f, epsilon, args=()):
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


class Matern_Quaternion(RBF):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu


    def __call__(self, X, Y=None, eval_gradient=False):

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            inner = 1- pdist(X, metric="cosine")
            X[inner<0]=-X[inner<0]
            inner = 1- pdist(X, metric="cosine")
            theta= np.arccos(2* inner** 2 -1)
            dists=theta/self.length_scale


        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            inner = 1- cdist(X,Y, metric="cosine")
            mask=inner<0
            X[mask[0]][:]=-X[mask[0]][:]
            inner = 1- cdist(X, Y, metric="cosine")
            theta= np.arccos(2* inner** 2 -1)
            dists=theta/self.length_scale    

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp**self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
            else:
                D = squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                denominator = np.sqrt(D.sum(axis=2))[:, :, np.newaxis]
                divide_result = np.zeros_like(D)
                np.divide(
                    D,
                    denominator,
                    out=divide_result,
                    where=denominator != 0,
                )
                K_gradient = K[..., np.newaxis] * divide_result
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu,
            )
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0], self.nu
            )



class Matern_SO3(RBF):
    def __init__(self, length_scale=np.ones(4), length_scale_bounds=(1e-5, 1e5), nu=1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu


    def __call__(self, X, Y=None, eval_gradient=False):

        X = np.atleast_2d(X)
        length_scale = self.length_scale
        if Y is None:
            X_quat=X[:,3:]
            X_lin=X[:,0:3]
            inner = 1 - pdist(X_quat, metric="cosine")
            X_quat[inner<0]=-X_quat[inner<0]
            inner = 1- pdist(X_quat, metric="cosine")
            theta= np.arccos(2* inner** 2 -1) #this may be redundant
            dist_quat=theta/self.length_scale[3]
            dist_lin = pdist(X_lin/length_scale[:3], metric="euclidean")
            dists=dist_quat+dist_lin

        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y_quat=Y[:,3:]
            Y_lin=Y[:,:3]
            X_quat=X[:,3:] 
            X_quat=X[:,:3] 
            inner = 1- cdist(X_quat,Y_quat, metric="cosine")
            mask=inner<0
            X_quat[mask[0]][:]=-X_quat[mask[0]][:]
            inner = 1- cdist(X_quat, Y_quat, metric="cosine")
            theta= np.arccos(2* inner** 2 -1)
            dist_quat=theta/self.length_scale[3]    
            dist_lin = cdist(X_lin/length_scale[:3], Y_lin/length_scale[:3], metric="euclidean")
            dists=dist_quat+dist_lin
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp**self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
            else:
                D = squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                denominator = np.sqrt(D.sum(axis=2))[:, :, np.newaxis]
                divide_result = np.zeros_like(D)
                np.divide(
                    D,
                    denominator,
                    out=divide_result,
                    where=denominator != 0,
                )
                K_gradient = K[..., np.newaxis] * divide_result
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu,
            )
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0], self.nu
            )            