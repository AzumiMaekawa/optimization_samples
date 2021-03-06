import numpy as np
import matplotlib.pyplot as plt
import copy


class Arm2D(object):
    def __init__(self):
        self.motorN = 2
        self.currentJointAngles = np.zeros(self.motorN)
        # self.currentJointAngles = np.random.uniform(
        # low=-np.pi, high=np.pi, size=self.motorN)
        self.linkLength = [0.8, 0.5]
        self.endEffectorPos = self.forward_kinematikcs(self.currentJointAngles)
        self.targetPos = self.endEffectorPos
        self.eps = 1e-3
        self.alpha = 0.5
        self.max_iterN = 50
        self.cnt = 0

    def forward_kinematikcs(self, jointAngles):
        # position x
        x = self.linkLength[0] * np.cos(jointAngles[0])\
            + self.linkLength[1] * np.cos(jointAngles[0] + jointAngles[1])
        # position y
        y = self.linkLength[0] * np.sin(jointAngles[0])\
            + self.linkLength[1] * np.sin(jointAngles[0] + jointAngles[1])
        # print('x, y: ', x, y)
        return np.array([x, y])  # return end effector's [x,y] position

    def inverse_kinematics_GadientDescent(self, targetPos, plot=False):
        self.targetPos = targetPos
        itr = 0
        while (1):
            print("targetPos:", self.targetPos)
            itr += 1
            err = self.objective_func(self.currentJointAngles)
            if itr > self.max_iterN:
                print('not reach in {}'.format(self.max_iterN))
                plt.ioff()
                self.plot_arm()
                return self.currentJointAngles
            if err < self.eps:
                print('reach at {} steps'.format(itr))
                plt.ioff()
                self.plot_arm()
                return self.currentJointAngles

            grad = self.calc_gradient(
                self.objective_func, self.currentJointAngles)
            # print('gradient: ', gradient)
            self.currentJointAngles = self.currentJointAngles - self.alpha * grad
            self.endEffectorPos = self.forward_kinematikcs(
                self.currentJointAngles)
            print("currentPos: ", self.endEffectorPos)
            if plot:
                plt.ion()
                self.plot_arm()

    def inverse_kinematics_NewtonsMethod(self, targetPos, plot=False):
        self.targetPos = targetPos
        itr = 0
        while (1):
            err = self.objective_func(self.currentJointAngles)
            itr += 1
            if itr > self.max_iterN:
                print('not reach in {} steps'.format(self.max_iterN))
                plt.ioff()
                self.plot_arm()
                return self.currentJointAngles
            if err < self.eps:
                print('reach at {} steps'.format(itr))
                plt.ioff()
                self.plot_arm()
                return self.currentJointAngles

            grad = self.calc_gradient(
                self.objective_func, self.currentJointAngles)
            hessian = self.calc_hessian(
                self.objective_func, self.currentJointAngles)
            # the Hessian regularization (this seems to be important for convergence)
            hessian += 0.5 * np.eye(hessian.shape[0])
            delta_x = np.linalg.solve(hessian, grad)
            self.currentJointAngles = self.currentJointAngles - self.alpha * delta_x
            self.endEffectorPos = self.forward_kinematikcs(
                self.currentJointAngles)
            print("currentPos: ", self.endEffectorPos)
            if plot:
                plt.ion()
                self.plot_arm()

    def calc_hessian(self, f, x):
        h = 1e-5
        hessian = np.eye(x.size)

        for row in range(x.size):
            for col in range(x.size):
                store_x = copy.deepcopy(x)

                # ret = self.calc_gradient(self.calc_gradient(f, x), x)
                # print("ret: ", ret)
                # f(x1+h, x2+h)
                x[row] += h
                x[col] += h
                f_x_pp_h = f(x)
                x = copy.deepcopy(store_x)

                # f(x1+h, x2-h)
                x[row] += h
                x[col] -= h
                f_x_pm_h = f(x)
                x = copy.deepcopy(store_x)

                # f(x1-h, x2+h)
                x[row] -= h
                x[col] += h
                f_x_mp_h = f(x)
                x = copy.deepcopy(store_x)

                # f(x1-h, x2-h)
                x[row] -= h
                x[col] -= h
                f_x_mm_h = f(x)

                hessian[row][col] = (
                    f_x_pp_h - f_x_pm_h - f_x_mp_h + f_x_mm_h) / (4 * h)

        return hessian

    def objective_func(self, jointAngles):
        err_vec = self.forward_kinematikcs(jointAngles) - self.targetPos
        ret_val = (1./2.) * np.dot(err_vec, err_vec)
        # print('loss: ', ret_val)
        return ret_val

    # bad example
    def grad_function(self, f, x):
        h = 1e-4 * np.ones(self.motorN)
        return (f(x + h) - f(x - h)) / (2. * h)

    def calc_gradient(self, f, x):
        h = 1e-4
        gradient = np.zeros_like(x)

        for i in range(x.size):
            store_x = x[:]

            # f(x+h)
            x[i] += h
            f_x_plus_h = f(x)
            x = store_x[:]

            # f(x-h)
            x[i] -= h
            f_x_minus_h = f(x)

            gradient[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)

        return gradient

    def plot_arm(self):
        self.cnt += 1
        dt = 1e-8
        shoulder = np.array([0, 0])
        elbow = shoulder + np.array(
            [self.linkLength[0] * np.cos(self.currentJointAngles[0]),
             self.linkLength[0] * np.sin(self.currentJointAngles[0])])
        wrist = self.endEffectorPos

        plt.cla()
        # 'k' means 'black color'
        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')

        plt.plot(self.targetPos[0], self.targetPos[1], 'bo', ms=10)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        plt.show()
        # plt.savefig('opt-based-ik-fig/img{}.png'.format(self.cnt))
        plt.pause(dt)


def main():
    arm2d = Arm2D()
    # ion means Interactive mode ON. this seems to need for animation
    # plt.ion()  # ion means Interactive mode ON. this seems to need for animation
    # arm2d.forward_kinematikcs([np.pi, -np.pi/4.])
    # arm2d.inverse_kinematics_GadientDecsent([-1., -0.5], plot=True)
    arm2d.inverse_kinematics_NewtonsMethod([-1., 0.5], plot=True)
    # plt.ioff()


if __name__ == "__main__":
    main()
