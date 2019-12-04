#!/usr/bin/env python

import numpy as np
from scipy.optimize import fmin_cg
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import pairwise_distances


class ModelBasedRecommender():
    def __init__(self, lambda_=0.1, n_features=20):
        self.lambda_ = lambda_
        self.n_features = n_features

    def fit(self, Y):
        # Computes the user and item feature vectors. They are stored in self.Theta_ and self.X_ respectively
        self.Y = Y
        n_items, n_users = Y.shape
        self.X_ = np.random.rand(n_items, self.n_features)
        self.Theta_ = np.random.rand(n_users, self.n_features)
        # X and Theta are squeezed into a 1-d array, for the minimizer
        params = np.vstack((self.X_, self.Theta_)).ravel()
        # fmin_cg minimizes a function using the gradient starting from a point (params) until convergence (or 100 iterations)
        # In this case, the starting point is randomly defined. In the case of updating the model when adding a user (recomputing self.X_ and self.Theta_), you can use the old self.X_ and self.Theta_
        # as the starting point and the minimizer should converge faster
        solution = fmin_cg(model_cost, params, fprime=model_cost_gradient, args=(Y, self.n_features, self.lambda_),
                           maxiter=100, disp=False)
        # X and Theta are restored from the params
        self.X_ = solution[:n_items * self.n_features].reshape(Y.shape[0], self.n_features)
        self.Theta_ = solution[n_items * self.n_features:].reshape(Y.shape[1], self.n_features)
        return self

    def predict_many(self, items, users):
        # Predicts the rating of all the pairs (item, user) from the items and users variables
        # The items and users variables have to be the same size so that:
        # predict_many(np.array([0, 1]), np.array([3, 4])) = np.array([predict(0, 3), predict(1, 4)])
        pre_result = self.X_.dot(self.Theta_.T)
        return np.rint(np.clip(pre_result[items, users], 1, 5))

    def predict(self, item, user):
        # Predicts the rating of item from user
        return self.predict_many(item, user)


def model_cost(params, Y, n_features, lambda_):
    # Computes the model error
    X = params[:Y.shape[0] * n_features].reshape(Y.shape[0], n_features)
    Theta = params[Y.shape[0] * n_features:].reshape(Y.shape[1], n_features)
    estimate = X.dot(Theta.T) * (Y != 0)
    result = np.sum(np.square(estimate - Y))
    result = result / 2.0 + lambda_ / 2.0 * np.sum(np.square(X)) + lambda_ / 2.0 * np.sum(np.square(Theta))
    return result


def model_cost_gradient(params, Y, n_features, lambda_):
    # Computes the gradient of model error
    X = params[:Y.shape[0] * n_features].reshape(Y.shape[0], n_features)
    Theta = params[Y.shape[0] * n_features:].reshape(Y.shape[1], n_features)
    temp_value = (X.dot(Theta.T) - Y) * (Y != 0)
    X_grad = temp_value.dot(Theta) + lambda_ * X
    Theta_grad = temp_value.T.dot(X) + lambda_ * Theta
    return np.vstack((X_grad, Theta_grad)).ravel()


class ModelMean():
    # Para que quede igual que el anterior :)
    def __init__(self):
        return

    def fit(self, Y):
        self.Y = Y
        self.n_items, self.n_users = Y.shape
        # print('Fit Model Mean')
        # print('Nro items: ' + str(self.n_items))
        # print('Nro users: ' + str(self.n_users))
        return self

    def predict(self, item, user):
        # si me entregan una pelicula o un usuario que no existe
        if ((item > self.n_items) | (user > self.n_users)):
            return 0.0
        items = self.Y[:, user]  # vector con los items evaluados por el user
        users = self.Y[item, :]  # vector con los users que evaluaron el item

        items = items[np.nonzero(items)]  # eliminar los items no evaluados por el user
        users = users[np.nonzero(users)]  # eliminar los usuarios que no evaluaron el item

        # Verificar si el usuario ha evaluado por lo menos algun item
        items_mean = 0.0
        if items.size > 0:
            items_mean = np.mean(items)
        # Verificar si el item ha sido evaluado por al menos un usuario
        users_mean = 0.0
        if users.size > 0:
            users_mean = np.mean(users)
        # Correccion del factor en caso de inexistencia de evaluaciones del usuario o del item
        factor = 0.5
        if (items.size == 0 | users.size == 0):
            factor = 1.0

        return factor * (items_mean + users_mean)

    # pre_result es un vector que entrega la evaluacion predicha para cada par item, user
    def predict_many(self, items, users):
        pre_result = np.zeros_like(items)

        for i in range(pre_result.size):
            pre_result[i] = self.predict(items[i], users[i])

        return pre_result


class Model_KNN():
    def __init__(self, K=20):  # se tendra por defecto K=20
        self.K = K

    def fit(self, Y):
        self.Y = Y
        self.n_items, self.n_users = Y.shape
        self.sim = np.zeros((self.n_items, self.n_items))

        # Construir vector de evaluacion media por cada usuario
        users_mean = np.zeros(self.n_users)
        for user in range(self.n_users):
            items = self.Y[:, user]  # vector con los items evaluados por el user
            items = items[np.nonzero(items)]  # eliminar los items no evaluados por el user
            mean = 0.0
            # Verificar si el usuario ha evaluado por lo menos algun item
            if items.size > 0:
                mean = np.mean(items)
            users_mean[user] = mean

        # Construir medida de similitud entre peliculas (correlacion de Pearson)
        for i in range(self.n_items):
            print (str(i) + ' de 1682')  # porcentaje de avance en el calculo de la matriz
            for j in range(self.n_items):
                suma_1 = 0.0
                suma_2 = 0.0
                suma_3 = 0.0
                for k in range(self.n_users):
                    suma_1 = suma_1 + (Y[i, k] - users_mean[k]) * (Y[j, k] - users_mean[k])
                    suma_2 = suma_2 + (Y[i, k] - users_mean[k]) * (Y[i, k] - users_mean[k])
                    suma_3 = suma_3 + (Y[j, k] - users_mean[k]) * (Y[j, k] - users_mean[k])

                if (suma_2 == 0.0 or suma_3 == 0.0):
                    self.sim[i, j] = 0.0
                else:
                    self.sim[i, j] = suma_1 / (np.sqrt(suma_2) * np.sqrt(suma_3))

        np.savetxt('sim_matrix.txt', self.sim)

        return self

    def fit_2(self, Y):
        self.Y = Y
        self.n_items, self.n_users = Y.shape
        self.sim = np.loadtxt('sim_matrix.txt')


    def predict(self, item, user):
        # si me entregan una pelicula o un usuario que no existe
        if ((item > self.n_items) | (user > self.n_users)):
            return 0.0
        sim_k = self.sim[item, :]  # vector de similitud para la pelicula "item"
        sorted_idx = np.argsort(-sim_k)  # indica los indices que entregan las mayores similitudes

        suma_1 = 0.0
        suma_2 = 0.0

        contador = 0
        for k in range(self.n_items):
            index_k = sorted_idx[k]
            if Y[index_k, user] > 0:
        #al buscar las peliculas mas similares, puede que esas no las haya evaluado el usuario j, por lo que no se puede estimar de buena forma la evaluacion de j a la pelicula i
        #si el usuario j no ha evaluado ninguna de las 20 peliculas mas similares la prediccion sera 0
                contador = contador + 1
                suma_1 = suma_1 + self.Y[index_k, user] * self.sim[item, index_k]
                suma_2 = suma_2 + self.sim [item, index_k]
            if contador == self.K:
                break

        if suma_2 == 0.0:
            return 0.0   #para que no queden denominadores 0
        else:
            return suma_1/suma_2





    # pre_result es un vector que entrega la evaluacion predicha para cada par item, user

    def predict_many(self, items, users):
        pre_result = np.zeros_like(items)

        for i in range(pre_result.size):
            pre_result[i] = self.predict(items[i], users[i])

        return pre_result
