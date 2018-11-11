import dubins
import matplotlib.pyplot as plt
import numpy as np
import os

class DubinsPark:
    ''' Class to generate paths using Dubins Curves with randomized 
        initial/goal positions and orientations
        Journey McDowell (c) 2018 
    '''
    def __init__(self, turning_radius, step_size):
        self.turning_radius = turning_radius
        self.step_size = step_size

    def distance(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_menger_curvature(self, a, b, c):
        ''' method to get curvature from three points '''
        raw_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        triangle_area = raw_area / 2.0

        return 4 * triangle_area / (self.distance(a, b) 
               * self.distance(b, c) * self.distance(c, a))

    def generate(self):
        ## Randomize starting and goal poses
        x0 = np.random.randint(-200, 200)
        y0 = np.random.randint(-200, 200)
        psi_0 = np.radians(np.random.randint(0, 360))

        x1 = np.random.randint(-200, 200)
        y1 = np.random.randint(-200, 200)
        psi_1 = np.radians(np.random.randint(0, 360))

        self.q0 = [x0, y0, psi_0]
        self.qg = [x1, y1, psi_1]

        ## Modify dubins to work for a straight offset from goal
        self.q1 = self.qg.copy()
        self.q1[0] -= self.turning_radius * np.cos(self.q1[2])
        self.q1[1] -= self.turning_radius * np.sin(self.q1[2])

        # Dubins
        path = dubins.shortest_path(self.q0, self.q1, self.turning_radius)
        qs, dist_dubins = path.sample_many(self.step_size)
        qs = np.array(qs)

        ## Concatenate with reverse straight
        s_s = self.distance((self.q1[0], self.q1[1]), (self.qg[0], self.qg[1]))
        n_steps = int(s_s // self.step_size) + 1
        straight = np.array([np.linspace(self.q1[0], self.qg[0], n_steps), 
                            np.linspace(self.q1[1], self.qg[1], n_steps), 
                            self.qg[2] * np.ones(n_steps)]).T
        qs = np.vstack((qs, straight))

        dist_straight = [dist_dubins[-1]]
        for j in range(len(straight)):
            dist_straight.append(dist_straight[j] + (s_s / n_steps))
        self.dist = dist_dubins + dist_straight[1:] # ignore double counting

        ## x, y, curv, psi, dist
        self.curv = []
        for n in range(len(qs)):
            if n == 0:
                self.curv.append(self.get_menger_curvature(qs[0], qs[n+1], qs[n+2]))
            elif n == len(qs) - 1:
                self.curv.append(self.get_menger_curvature(qs[n-2], qs[n-1], qs[n]))
            else:
                self.curv.append(self.get_menger_curvature(qs[n-1], qs[n], qs[n+1]))

        self.x = qs[:, 0]
        self.y = qs[:, 1]
        self.psi = qs[:, 2]

    def display(self, plot_number=None):
        ## Plot again to make sure
        plt.plot(self.x, self.y)
        plt.quiver(self.x, self.y, np.cos(self.psi), np.sin(self.psi), 0.5)
        plt.plot(self.qg[0], self.qg[1], 'x')
        plt.axis('equal')
        plt.xlabel('Position in x [m]')
        plt.ylabel('Position in y [m]')
        plt.title('Plot #{}'.format(plot_number))
        plt.show()

    def send_to_txt(self, plot_number=None):
        if not os.path.exists('./dubins_path'):
            os.mkdir('./dubins_path')
        with open('./dubins_path/dubins_path_{}.txt'.format(plot_number), 'a') as file:
            for i in range(len(self.x)):
                file.write(str(self.x[i]) + ',' + str(self.y[i]) + ',' + 
                           str(self.curv[i]) + ',' + str(self.psi[i]) + ',' + 
                           str(self.dist[i]) + '\n')
