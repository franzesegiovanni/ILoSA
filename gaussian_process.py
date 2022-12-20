"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#!/usr/bin/env python
import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


class InteractiveGP():
    def __init__(self, X, Y, kernel, y_lim, alpha=1e-10, n_restarts_optimizer=20):
        self.y_lim=y_lim
        self.X=np.transpose(X)
        print("Shape X for gaussian Process")
        print(np.shape(self.X))
        self.Y=np.transpose(Y)
        print("Shape Y for gaussian Process")
        print(np.shape(self.Y))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)

    
    def fit(self):

        gp_ = self.gp.fit(self.X, self.Y)

        self.kernel_ = gp_.kernel_

        self.length_scales=self.kernel_.get_params()['k1__k2__length_scale']

        self.noise_var_ = gp_.alpha + self.kernel_.get_params()['k2__noise_level']

        self.max_var   = self.kernel_.get_params()['k1__k1__constant_value']+ self.noise_var_

        K_ = self.kernel_(self.X, self.X) + (self.noise_var_ * np.eye(len(self.X)))

        self.K_inv = np.linalg.inv(K_)

    def predict(self, x, return_std=True):
        k_star = self.kernel_(self.X, x).reshape(-1, 1)
        k_star_K_inv_ = np.matmul(np.transpose(k_star), self.K_inv)
        self.mu=np.matmul(k_star_K_inv_, self.Y)
        self.sigma=None
        if return_std==True:
            self.sigma = np.subtract(self.kernel_(x, x)+ self.noise_var_, np.matmul(k_star_K_inv_, k_star))
        return self.mu, self.sigma 

    def var_gradient(self, x):
        lscale=self.length_scales
        k_star = self.kernel_(self.X, x).reshape(-1, 1)
        dKdx = 2* k_star * (self.X[:, 0].reshape(-1, 1) - x[0][0]) / (lscale[0] ** 2) 
        dKdy = 2* k_star * (self.X[:, 1].reshape(-1, 1) - x[0][1]) / (lscale[1] ** 2)
        dKdz = 2* k_star * (self.X[:, 2].reshape(-1, 1) - x[0][2]) / (lscale[2] ** 2)
        a = - 2 * np.matmul(np.transpose(k_star), self.K_inv)
        dSigma_dx_ = np.matmul(a, np.reshape(dKdx, [len(a[0]), 1]))
        dSigma_dy_ = np.matmul(a, np.reshape(dKdy, [len(a[0]), 1]))
        dSigma_dz_ = np.matmul(a, np.reshape(dKdz, [len(a[0]), 1]))

        return float(dSigma_dx_), float(dSigma_dy_), float(dSigma_dz_)


    def update_K_inv(self,x):
        """
        Van Vaerenbergh, Steven, et al. 
        "Fixed-budget kernel recursive least-squares." 
        2010 IEEE International Conference on Acoustics, Speech and Signal Processing. IEEE, 2010.
        """

        k_star_ = self.kernel_(self.X, x)
        B=k_star_.reshape(-1,1)
        C=np.transpose(B)
        D=self.kernel_(x.reshape(1,-1),x.reshape(1,-1))+ self.noise_var_
        A_inv_B=np.matmul(self.K_inv,B)
        K_inv_22=1/(D-np.matmul(C,A_inv_B))
        K_inv_12=-A_inv_B*K_inv_22
        K_inv_21=np.transpose(K_inv_12)
        K_inv_11=np.add(self.K_inv,np.matmul(A_inv_B,-K_inv_21))
        self.K_inv=np.block([
        [K_inv_11, K_inv_12 ],
        [K_inv_21, K_inv_22 ]])


    # Update changes made to the attractor
    # training_data_ - data used to train the GPs 2x1 format [X; Y]
    # correction_info_ - vector of the format [x; mu; epsilon_mu]
    # val_lim - 2x1 vector which contains the minimum and maximum bounds of the corrected value
    def update_with_A(self, x, mu, epsilon_mu, is_uncertain):
        """
        Franzese, Giovanni, et al. "ILoSA: Interactive learning of stiffness and attractors." 
        2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021.
        """
        x = x.reshape(1, -1)
        if is_uncertain:
            # region of uncertainty - add new point
            corrected_output=np.add(mu, np.array(epsilon_mu).reshape(1, -1))
            corrected_output = np.clip(corrected_output, self.y_lim[0], self.y_lim[1]) 
            self.Y = np.concatenate((self.Y, corrected_output), axis=0)
            self.update_K_inv(x)
            self.X = np.concatenate((self.X, x), axis=0)
        else:
            # known region - correct existing data
            corrected_output = np.array(epsilon_mu).reshape(1, -1)
            k_star_ = self.kernel_(self.X, x)
            A_inv_ = np.clip(scipy.linalg.pinv(np.matmul(np.transpose(k_star_), self.K_inv)), -1, 1)
            self.Y= np.clip((self.Y+ (corrected_output * A_inv_)).reshape(-1, np.size(mu)), self.y_lim[0], self.y_lim[1]) 


    # Update changes made to the attractor
    # training_data_ - data used to train the GPs 2x1 format [X; Y]
    # correction_info_ - vector of the format [x; mu; epsilon_mu]
    # val_lim - 2x1 vector which contains the minimum and maximum bounds of the corrected value
    def update_with_k(self, x, mu, epsilon_mu, is_uncertain):
        """
        Meszaros, Anna, Giovanni Franzese, and Jens Kober. 
        "Learning to Pick at Non-Zero-Velocity from Interactive Demonstrations." 
        IEEE Robotics and Automation Letters (2022).
        """
        if is_uncertain:
            # region of uncertainty - add new point
            corrected_output = np.clip(np.add(mu, epsilon_mu).reshape(1, -1), self.y_lim[0], self.y_lim[1]) 
            self.Y = np.concatenate((self.Y, corrected_output), axis=0)
            self.update_K_inv(x)
            self.X = np.concatenate((self.X, x), axis=0)
        else:
            # known region - correct existing data
            x= x.reshape(1, -1) 
            corrected_output = np.array(epsilon_mu).reshape(1, -1) 
            k_star_ = np.divide(self.kernel_(self.X, x), self.max_var )
            self.Y = np.clip((self.Y + (corrected_output * k_star_)).reshape(-1, np.size(mu)), self.y_lim[0], self.y_lim[1]) 

    def is_uncertain(self,theta):
        uncertain = (self.sigma - self.noise_var_)/(self.max_var - self.noise_var_) > theta
        return uncertain

