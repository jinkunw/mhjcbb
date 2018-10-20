import numpy as np
import gtsam

from sim import *


def X(x):
    return gtsam.symbol(ord('x'), x)


def L(l):
    return gtsam.symbol(ord('l'), l)


def dr(sim):
    traj_est = []
    landmark_est = []
    for x, (odom, obs) in enumerate(sim.step()):
        if x == 0:
            traj_est.append(odom)
        else:
            traj_est.append(traj_est[-1].compose(odom))
        for l, br in obs.items():
            point = gtsam.Point2(br[1] * np.cos(br[0]), br[1] * np.sin(br[0]))
            landmark_est.append((l, traj_est[-1].transform_from(point)))
    traj_est = np.array([(p.x(), p.y(), p.theta()) for p in traj_est])
    landmark_est = np.array([(p.x(), p.y(), l) for l, p in landmark_est])
    return traj_est, landmark_est


def slam(sim, da='jcbb', prob=0.95):
    isam2 = gtsam.ISAM2()
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    observed = set()
    l_da_true = {}
    for x, (odom, obs) in enumerate(sim.step()):
        if x == 0:
            prior_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_x, sim.sigma_y, sim.sigma_theta]))
            prior_factor = gtsam.PriorFactorPose2(X(0), odom, prior_model)
            graph.add(prior_factor)
            values.insert(X(0), odom)
        else:
            odom_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_x, sim.sigma_y, sim.sigma_theta]))
            odom_factor = gtsam.BetweenFactorPose2(X(x - 1), X(x), odom, odom_model)
            graph.add(odom_factor)
            pose0 = isam2.calculateEstimatePose2(X(x - 1))
            values.insert(X(x), pose0.compose(odom))

        isam2.update(graph, values)
        graph.resize(0)
        values.clear()

        if da == 'perfect':
            for l, br in obs.items():
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
                br_factor = gtsam.BearingRangeFactor2D(X(x), L(l), gtsam.Rot2(br[0]), br[1], br_model)
                graph.add(br_factor)
                if l not in observed:
                    pose1 = isam2.calculateEstimatePose2(X(x))
                    point = gtsam.Point2(br[1] * np.cos(br[0]), br[1] * np.sin(br[0]))
                    values.insert(L(l), pose1.transform_from(point))
                    observed.add(l)
        elif da == 'jcbb':
            jcbb = gtsam.da_JCBB2(isam2.getFactorsUnsafe(), isam2.calculateEstimate(), prob)
            for l, br in obs.items():
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
                jcbb.add(gtsam.Rot2(br[0]), br[1], br_model)
            keys = jcbb.match()
            keys = [gtsam.symbolIndex(keys.at(i)) for i in range(keys.size())]
            for (l_true, br), l in zip(obs.items(), keys):
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
                l_da_true[l] = l_true

                br_factor = gtsam.BearingRangeFactor2D(X(x), L(l), gtsam.Rot2(br[0]), br[1], br_model)
                graph.add(br_factor)
                if l not in observed:
                    pose1 = isam2.calculateEstimatePose2(X(x))
                    point = gtsam.Point2(br[1] * np.cos(br[0]), br[1] * np.sin(br[0]))
                    values.insert(L(l), pose1.transform_from(point))
                    observed.add(l)

        isam2.update(graph, values)
        graph.resize(0)
        values.clear()

    marginals = gtsam.Marginals(isam2.getFactorsUnsafe(), isam2.calculateEstimate())
    traj_est = [isam2.calculateEstimatePose2(X(x)) for x in range(len(sim.traj))]
    landmark_est = [(l, isam2.calculateEstimatePoint2(L(l))) for l in observed]
    traj_est = np.array([(p.x(), p.y(), p.theta()) for p in traj_est])
    if da == 'perfect':
        landmark_est = np.array([(p.x(), p.y(), l) for l, p in landmark_est])
    else:
        landmark_est = np.array([(p.x(), p.y(), l_da_true[l]) for l, p in landmark_est])
    return traj_est, landmark_est


def plot_sim(ax, sim):
    traj = np.array([(p.x(), p.y(), p.theta()) for p in sim.traj])
    landmark = np.array([(p.x(), p.y(), l) for l, p in sim.env.items()])
    ax.scatter(landmark[:, 0], landmark[:, 1], marker='x', c=landmark[:, 2], lw=2, s=100, label='Landmark - GT')
    ax.plot(traj[:, 0], traj[:, 1], 'k-', lw=2, label='Trajectory - GT')
    ax.set_aspect('equal')
    ax.legend()


def plot_est(ax, traj_est, landmark_est):
    ax.scatter(landmark_est[:, 0], landmark_est[:, 1], c=landmark_est[:, 2], marker='+', lw=2, alpha=0.5,
               label='Landmark - Est')
    ax.plot(traj_est[:, 0], traj_est[:, 1], 'go-', lw=2, label='Trajectory - Est')
    ax.set_aspect('equal')
    ax.legend()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Design a simulator
    sim = Simulator()
    sim.random_map(100, (-10, 10, -10, 10))
    pose = gtsam.Pose2(5, -5, np.pi / 2.0)
    for i in range(50):
        sim.traj.append(pose)
        if i % 10 == 0 and i > 0:
            u = 0.0, 1.0, np.pi / 2.0
        else:
            u = 1.0, 0.0, 0.0
        pose = pose.compose(gtsam.Pose2(*u))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, True, True)

    # Dead-reckoning
    sim.reset()
    traj_est, landmark_est = dr(sim)
    plot_sim(ax1, sim)
    plot_est(ax1, traj_est, landmark_est)
    ax1.set_title('Odometry')

    # Perfect DA
    sim.reset()
    traj_est, landmark_est = slam(sim, 'perfect')
    plot_sim(ax2, sim)
    plot_est(ax2, traj_est, landmark_est)
    ax2.set_title('Perfect DA')

    # JCBB
    sim.reset()
    traj_est, landmark_est = slam(sim, 'jcbb')
    plot_sim(ax3, sim)
    plot_est(ax3, traj_est, landmark_est)
    ax3.set_title('JCBB')

    plt.tight_layout()
    plt.show()
