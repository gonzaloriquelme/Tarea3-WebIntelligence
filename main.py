#!/usr/bin/env python

import numpy as np
from scipy.optimize import fmin_cg
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import os
from recommenders import *

os.chdir("/Users/iriqu/Desktop/Gonzalo/cursos importantes/Web int")

datafile = 'utility_matrix.txt'
mat = np.loadtxt(datafile)

x,y = np.nonzero(mat) #indices, x: pelicula, y: usuario

print('Numero de peliculas: ' + str(mat.shape[0]))
print('Numero de usuarios: ' + str(mat.shape[1]))
print('Numero de muestras no nulas: ' + str(x.size))

np.random.seed(345) #para que el random sea el mismo (para el caso de KNN y la matriz de similitud)
indices = np.random.permutation(x.size)

i_train = indices[:int(3.0*x.size/4)]
i_test = indices[int(3.0*x.size/4):]

Y_train = np.zeros_like(mat)
Y_train[x[i_train],y[i_train]] = mat[x[i_train],y[i_train]]

## Model Mean
print ('Model Mean ...')
model_mean = ModelMean()
model_mean.fit(Y_train)
print ('estamos ready con el fit')


pred_mean = model_mean.predict_many(x[i_test],y[i_test])
error_mean = np.sqrt(mean_squared_error(mat[x[i_test],y[i_test]],pred_mean))

print('Error Model Mean: ' + str(error_mean))

## Model Based Recommender
print ('Model Based Recommender ...')
model_recommender = ModelBasedRecommender()
model_recommender.fit(Y_train)
print ('estamos ready con el fit')


pred_rec = model_recommender.predict_many(x[i_test],y[i_test])
error_rec = np.sqrt(mean_squared_error(mat[x[i_test],y[i_test]],pred_rec))

print('Error Model Recommender: ' + str(error_rec))



#Se sugiere comentar el Model KNN en caso de correr multiples veces los modelos anteriores
## Model KNN
print ('Model KNN ...')
model_KNN = Model_KNN(20)
model_KNN.fit (Y_train)  #se comenta luego del calculo de la matriz de similitud
print ('estamos ready con el fit')
#model_KNN.fit_2 (Y_train)   #se utiliza luego del calculo de la matriz de similitud

pred_KNN = model_KNN.predict_many(x[i_test],y[i_test])
error_KNN = np.sqrt(mean_squared_error(mat[x[i_test],y[i_test]],pred_KNN))

print('Error Model KNN: ' + str(error_KNN))

