import numpy as np
import gtsam


class Simulator(object):
    def __init__(self, seed=0):
        self.sigma_x = 0.1
        self.sigma_y = 0.1
        self.sigma_theta = 0.05
        self.sigma_bearing = 0.05
        self.sigma_range = 0.1

        self.max_range = 5.0
        self.max_bearing = np.pi / 3.0
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        # Define env and traj here
        self.env = {}  # l -> gtsam.Point2
        self.traj = []  # gtsam.Pose2

    def reset(self):
        self.random_state = np.random.RandomState(self.seed)

    def random_map(self, size, limit):
        """
        size: num of landmarks
        limit: l, r, b, t
        """
        self.env = {}
        l, r, b, t = limit
        for i in range(size):
            x = self.random_state.uniform(l, r)
            y = self.random_state.uniform(b, t)
            self.env[i] = gtsam.Point2(x, y)

    def step(self):
        """
        return:
          odom: odom between two poses (initial pose is returned for the first step)
          obs: dict of (landmark -> (bearing, range))
        """
        for i in range(len(self.traj)):
            if i == 0:
                odom = gtsam.Pose2()
            else:
                odom = self.traj[i - 1].between(self.traj[i])
            nx = self.random_state.normal(0.0, self.sigma_x)
            ny = self.random_state.normal(0.0, self.sigma_y)
            nt = self.random_state.normal(0.0, self.sigma_theta)
            odom = odom.compose(gtsam.Pose2(nx, ny, nt))

            obs = {}
            for l, point in self.env.items():
                b = self.traj[i].bearing(point).theta()
                r = self.traj[i].range(point)
                b += self.random_state.normal(0.0, self.sigma_bearing)
                r += self.random_state.normal(0.0, self.sigma_range)

                if 0 < r < self.max_range and abs(b) < self.max_bearing:
                    obs[l] = b, r

            if i == 0:
                yield self.traj[0].compose(odom), obs
            else:
                yield odom, obs
