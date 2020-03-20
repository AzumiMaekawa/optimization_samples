import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod


class ObjectiveFunction:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, x):
        pass

    def getGradient(self, x):  # x: numpy vector
        grad = np.zeros(x.size)
        self.addFiniteDifferenceGradientTo(x, grad)

    def addFiniteDifferenceGradientTo(self, x, grad):
        h = 1e-8
        for i in range(x.size):
            dx = np.zeros(x.size)
            dx[i] = h
            grad[i] = (self.evaluate(x + dx) -
                       self.evaluate(x - dx)) / (2. * h)


class QuadraticFunction(ObjectiveFunction):
    def __init(self):
        super().__init__()
        # for plot
        self.range = 5.
        self.levels = np.linspace(0.01, 30, num=10)

    def evaluate(self, x):
        return (1./2) * np.dot(x, x)


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        self.w1 = 1.
        self.w2 = 100.
        # for plot
        self.range = 4.
        self.levels = [0.1, 0.4, 1.6, 5.,
                       25., 125., 500., 1000., 2500]

    def evaluate(self, x):
        x_even = x[::2]
        x_odd = x[1::2]
        return np.dot(x_even - self.w1, x_even - self.w1) \
            + self.w2*np.dot(x_even**2-x_odd, x_even**2-x_odd)


class PlotContour:
    def __init__(self, objective):
        x_range = np.arange(-3, 3, 0.1)
        y_range = np.arange(-3, 3, 0.1)
        self.X, self.Y = np.meshgrid(x_range, y_range)
        self.objective = objective
        x = np.concatenate([[self.X], [self.Y]])
        self.Z = np.zeros(self.X.shape)
        for row in range(x.shape[1]):
            for col in range(x.shape[2]):
                self.Z[row, col] = self.objective.evaluate(x[:, row, col])

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x", size=16)
        ax.set_ylabel("y", size=16)
        ax.set_zlabel("z", size=16)

        ax.plot_surface(self.X, self.Y, self.Z)

        plt.show()

    def plot2D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel("x", size=16)
        ax.set_ylabel("y", size=16)

        # contour plot
        im = ax.contourf(self.X, self.Y, self.Z, levels=objective.levels,
                         cmap="summer", alpha=0.5)
        # color map plot
        # im = ax.pcolormesh(self.X, self.Y, self.Z, cmap="Blues")
        fig.colorbar(im)

        plt.show()


class Minimizer:
    def __init__(self):
        pass


if __name__ == '__main__':
    # objective = QuadraticFunction()
    objective = RosenbrockFunction()
    plot = PlotContour(objective)

    # plot.plot3D()
    plot.plot2D()
    # plotter()
