#!/usr/bin/env python3
""" doc """
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """ doc """
    def __init__(self, nx):
        """ doc """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ doc """
        return self.__W

    @property
    def b(self):
        """ doc """
        return self.__b

    @property
    def A(self):
        """ doc """
        return self.__A

    def forward_prop(self, X):
        """ doc """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """ doc """
        m = Y.shape[1]
        cost = (-(1/m)) * np.sum((Y * np.log(A)) +
                                 ((1 - Y) * np.log(1.0000001 - A)))
        return (cost)

    def evaluate(self, X, Y):
        """ doc """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.round(A).astype(int), cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ doc """
        m = X.shape[1]
        dz = A - Y
        db = (1 / m) * np.sum(dz)
        dw = (1 / m) * np.matmul(X, dz.T)
        self.__W = self.__W - (alpha * dw).T
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ doc """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        cost_store = []
        iteration_store = []
        for i in range(iterations + 1):
            self.__A = self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            if verbose:
                if type(step) is not int:
                    raise TypeError("step must be an integer")
                if step < 0 and step > iterations:
                    raise ValueError("step must be positive and <= iterations")
                cost_store.append(cost)
                iteration_store.append(i)
                print("Cost after {} iterations: {}".format(i, cost))
                verbose = False
                graph = False
            self.gradient_descent(X, Y, self.__A, alpha)
            if ((i + 1) % step) == 0:
                verbose = True
            if i == iterations:
                graph = True
            if graph:
                if type(step) is not int:
                    raise TypeError("step must be an integer")
                if step < 0 and step > iterations:
                    raise ValueError("step must be positive and <= iterations")
                plt.plot(iteration_store, cost_store, color='blue')
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")
                plt.show()
        return self.evaluate(X, Y)
