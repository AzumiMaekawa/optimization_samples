import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from abc import abstractmethod


class ObjectiveFunction:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, x):
        pass

    def getGradient(self, x):  # x: numpy vector
        # grad = np.zeros(x.size)
        grad = self.addFiniteDifferenceGradient(x)
        return grad

    def getHessian(self, x):
        hessian = self.addFiniteDifferenceHessian(x)
        return hessian

    def addFiniteDifferenceGradient(self, x):
        h = 1e-5
        grad = np.zeros(x.size)
        for i in range(x.size):
            dx = np.zeros(x.size)
            dx[i] = h
            fp = self.evaluate(x + dx)
            fm = self.evaluate(x - dx)
            grad[i] = (fp - fm) / (2. * h)

        return grad

    def addFiniteDifferenceHessian(self, x):
        h = 1e-5
        hessian = np.zeros((x.size, x.size))
        for i in range(x.size):
            dx = np.zeros(x.size)
            dx[i] = h
            gp = self.getGradient(x + dx)
            gm = self.getGradient(x - dx)
            hess = (gp - gm) / (2. * h)
            for j in range(x.size):
                if abs(hess[j]) > 1e-10:
                    hessian[i, j] = hess[j]
        return hessian


class QuadraticFunction(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        # for plot
        self.range = 5.
        self.levels = np.linspace(0, 30, num=10)

    def evaluate(self, x):
        x = x.reshape(x.size,)
        return (1./2) * np.dot(x, x)


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self):
        super().__init__()
        self.w1 = 1.
        self.w2 = 100.
        # for plot
        self.range = 4.
        self.levels = [0, 0.4, 1.6, 5.,
                       25., 125., 500., 1000., 2500]

    def evaluate(self, x):
        x = x.reshape(x.size, )
        x_even = x[::2]
        x_odd = x[1::2]
        return np.dot(x_even - self.w1, x_even - self.w1) \
            + self.w2*np.dot(x_even**2-x_odd, x_even**2-x_odd)


class PlotContour:
    def __init__(self, objective):
        axis_range = objective.range
        x_range = np.arange(-axis_range, axis_range, 0.1)
        y_range = np.arange(-axis_range, axis_range, 0.1)
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
        im = ax.contourf(self.X, self.Y, self.Z, levels=self.objective.levels,
                         cmap="summer", alpha=0.5)
        # color map plot
        # im = ax.pcolormesh(self.X, self.Y, self.Z, cmap="Blues")
        fig.colorbar(im)

        plt.show()

    def plot2dWithTrajectory(self, x_traj):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel("x", size=16)
        ax.set_ylabel("y", size=16)

        im = ax.contourf(self.X, self.Y, self.Z,
                         levels=objective.levels, cmap="summer", alpha=0.5)
        fig.colorbar(im)

        # plot trajectoy
        ax.plot(x_traj[0, :], x_traj[1, :],
                linewidth=0.5, marker='o', markersize=4, zorder=1)
        ax.scatter(x_traj[0, 0], x_traj[1, 0], color='m', marker='*',
                   s=18, label='initial', zorder=2)
        ax.scatter(x_traj[0, -1], x_traj[1, -1],
                   color='r', marker='*', s=18, label='finish', zorder=2)
        ax.legend()
        plt.show()


class Minimizer:
    def __init__(self):
        self.x_trajectory = None
        self.evaluation_trajectory = []
        self.lastIterations = 0

    @abstractmethod
    def minimize(self, objective, x):
        pass

    def reset(self):
        self.x_trajectory = None
        self.evaluation_trajectory = []
        self.lastIterations = 0


class GradientDescent(Minimizer):
    def __init__(self):
        super().__init__()
        self.stepSize = 0.01
        self.maxIteration = 50
        self.conversionThreshold = 1e-10

    def minimize(self, objective, x):
        isConverged = False
        self.x_trajectory = x.reshape(x.size, 1)
        for i in range(self.maxIteration):
            dx = self.computeSearchDirection(objective, x)
            if np.linalg.norm(dx, ord=2) < self.conversionThreshold:
                isConverged = True
                self.lastIterations = i
                break
            elif i == self.maxIteration - 1:
                print("reach max iteration")
                self.lastIterations = self.maxIteration

            x = self.step(objective, x)
            # x = x.reshape(x.size, 1)
            self.x_trajectory = np.append(
                self.x_trajectory, x.reshape(x.size, 1), axis=1)
            self.evaluation_trajectory.append(objective.evaluate(x))

        return isConverged

    def computeSearchDirection(self, objective, x):
        dx = objective.getGradient(x)
        return dx

    def step(self, objective, x):
        dx = self.computeSearchDirection(objective, x)
        x -= self.stepSize * dx
        return x


class GradientDescentBackTrackingLineSearch(GradientDescent):
    def __init__(self):
        super().__init__()
        self.lineSearchMaxIteration = 15
        self.initialStepSize = 1.0
        self.scalingFactor = 0.5

    def step(self, objective, x):
        dx = self.computeSearchDirection(objective, x)
        for i in range(self.lineSearchMaxIteration):
            step_size = np.power(self.scalingFactor, i) * self.initialStepSize
            x_candidate = x - step_size * dx
            if objective.evaluate(x_candidate) < objective.evaluate(x):
                return x_candidate

        print("update failed....")
        return x


class NewtonsMethod(GradientDescentBackTrackingLineSearch):
    def __init__(self):
        super().__init__()
        self.reg = 0.

    def computeSearchDirection(self, objective, x):
        grad = objective.getGradient(x)
        hessian = objective.getHessian(x)
        identity_matrix = np.eye(x.size)
        hessian_inv = np.linalg.inv(hessian + self.reg * identity_matrix)
        dx = np.dot(hessian_inv, grad)
        return dx


def test(objective, x):
    h = 1e-8
    dx = np.zeros(2)
    dx[0] = h
    gp = objective.getGradient(x + dx)
    gm = objective.getGradient(x - dx)
    hess = (gp - gm) / (2. * h)
    hessian = objective.getHessian(x)
    print(gp)
    print(gm)
    print(hess)
    print(hessian)


if __name__ == '__main__':
    # objective = QuadraticFunction()
    objective = RosenbrockFunction()

    x = np.array([-1.2, -2.5])

    plot = PlotContour(objective)
    GDLS = GradientDescentBackTrackingLineSearch()
    Newton = NewtonsMethod()

    # do minimization
    GDLS.minimize(objective, x)
    Newton.minimize(objective, x)

    # plot the results
    plot.plot2dWithTrajectory(GDLS.x_trajectory)
    plot.plot2dWithTrajectory((Newton.x_trajectory))
    print("evaluate result  GDLS: {}, Newton: {}".format(
        GDLS.evaluation_trajectory[-1], Newton.evaluation_trajectory[-1]))
    print("iteration   GDLS: {}, Newton: {}".format(
        GDLS.lastIterations, Newton.lastIterations))
