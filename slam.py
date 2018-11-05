import numpy as np
import gtsam

from sim import *


def X(x):
    return gtsam.symbol(ord('x'), x)


def L(l):
    return gtsam.symbol(ord('l'), l)


def slam(sim, da='jcbb', prob=0.95):
    isam2 = gtsam.ISAM2()
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    observed = set()
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
        estimate = isam2.calculateEstimate()

        if da == 'dr':
            for l_true, br in obs.items():
                l = len(observed)
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
                br_factor = gtsam.BearingRangeFactor2D(X(x), L(l), gtsam.Rot2(br[0]), br[1], br_model)
                graph.add(br_factor)
                if l not in observed:
                    pose1 = isam2.calculateEstimatePose2(X(x))
                    point = gtsam.Point2(br[1] * np.cos(br[0]), br[1] * np.sin(br[0]))
                    values.insert(L(l), pose1.transform_from(point))
                    observed.add(l)
        elif da == 'perfect':
            for l_true, br in obs.items():
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
                br_factor = gtsam.BearingRangeFactor2D(X(x), L(l_true), gtsam.Rot2(br[0]), br[1], br_model)
                graph.add(br_factor)
                if l_true not in observed:
                    pose1 = isam2.calculateEstimatePose2(X(x))
                    point = gtsam.Point2(br[1] * np.cos(br[0]), br[1] * np.sin(br[0]))
                    values.insert(L(l_true), pose1.transform_from(point))
                    observed.add(l_true)
        elif da == 'jcbb':
            ################################################################################
            jcbb = gtsam.da_JCBB2(isam2, prob)
            for l, br in obs.items():
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
                jcbb.add(gtsam.Rot2(br[0]), br[1], br_model)

            keys = jcbb.match()
            ################################################################################

            keys = [gtsam.symbolIndex(keys.at(i)) for i in range(keys.size())]
            for (l_true, br), l in zip(obs.items(), keys):
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
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

    traj_est = [isam2.calculateEstimatePose2(X(x)) for x in range(len(sim.traj))]
    traj_est = np.array([(p.x(), p.y(), p.theta()) for p in traj_est])
    landmark_est = [isam2.calculateEstimatePoint2(L(l)) for l in observed]
    landmark_est = np.array([(p.x(), p.y()) for p in landmark_est])
    return [[traj_est, landmark_est]]


def prune1(slams, x, threshold):
    deleted = set()
    for i, (isam2_i, observed_i) in enumerate(slams):
        if i in deleted:
            continue
        pose_i = isam2_i.calculateEstimatePose2(X(x))
        cov_i = isam2_i.marginalCovariance(X(x))
        for j, (isam2_j, observed_j) in enumerate(slams):
            if j <= i or j in deleted:
                continue
            pose_j = isam2_j.calculateEstimatePose2(X(x))

            e = pose_i.localCoordinates(pose_j).reshape(-1, 1)
            if e.T.dot(np.linalg.inv(cov_i)).dot(e) < threshold**2:
                if len(observed_i) < len(observed_j):
                    deleted.add(j)
                else:
                    deleted.add(i)
                    break

    slams = [slams[i] for i in range(len(slams)) if i not in deleted]
    return sorted(slams, key=lambda x: x[1])


def prune2(slams, max_observed_diff=3):
    pruned = [slams[0]]
    for isam2, observed in slams[1:]:
        if len(observed) - len(slams[0][1]) < max_observed_diff:
            pruned.append([isam2, observed])
    return pruned


def mhjcbb(sim, num_tracks=10, prob=0.95, posterior_pose_md_threshold=1.5, prune2_skip=10, max_observed_diff=3):
    slams = [[gtsam.ISAM2(), set()]]

    prune2_count = 1
    observed = set()
    for x, (odom, obs) in enumerate(sim.step()):
        for isam2, observed in slams:
            graph = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()
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

        ################################################################################
        mhjcbb = gtsam.da_MHJCBB2(num_tracks, prob, posterior_pose_md_threshold)
        for isam2, observed, in slams:
            mhjcbb.initialize(isam2)

        for l, br in obs.items():
            br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
            mhjcbb.add(gtsam.Rot2(br[0]), br[1], br_model)

        mhjcbb.match()
        ################################################################################

        new_slams = []
        for i in range(mhjcbb.size()):
            track, keys = mhjcbb.get(i)
            keys = [gtsam.symbolIndex(keys.at(i)) for i in range(keys.size())]

            isam2 = gtsam.ISAM2()
            isam2.update(slams[track][0].getFactorsUnsafe(), slams[track][0].calculateEstimate())
            graph = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()
            observed = set(slams[track][1])
            for (l_true, br), l in zip(obs.items(), keys):
                br_model = gtsam.noiseModel_Diagonal.Sigmas(np.array([sim.sigma_bearing, sim.sigma_range]))
                br_factor = gtsam.BearingRangeFactor2D(X(x), L(l), gtsam.Rot2(br[0]), br[1], br_model)

                graph.add(br_factor)
                if l not in observed:
                    pose1 = isam2.calculateEstimatePose2(X(x))
                    point = gtsam.Point2(br[1] * np.cos(br[0]), br[1] * np.sin(br[0]))
                    values.insert(L(l), pose1.transform_from(point))
                    observed.add(l)
            isam2.update(graph, values)
            new_slams.append([isam2, observed])
        slams = new_slams
        slams = prune1(slams, x, posterior_pose_md_threshold)

        if len(slams[0][1]) > prune2_count * prune2_skip:
            slams = prune2(slams, max_observed_diff)
            prune2_count += 1

    result = []
    for isam2, observed in slams:
        traj_est = [isam2.calculateEstimatePose2(X(x)) for x in range(len(sim.traj))]
        traj_est = np.array([(p.x(), p.y(), p.theta()) for p in traj_est])
        landmark_est = [isam2.calculateEstimatePoint2(L(l)) for l in observed]
        landmark_est = np.array([(p.x(), p.y()) for p in landmark_est])
        result.append((traj_est, landmark_est))
    return result


def plot_sim(ax, sim):
    traj = np.array([(p.x(), p.y(), p.theta()) for p in sim.traj])
    landmark = np.array([(p.x(), p.y()) for p in sim.env.values()])
    p1 = ax.scatter(landmark[:, 0], landmark[:, 1], marker='x', lw=2, s=100)
    (p2, ) = ax.plot(traj[:, 0], traj[:, 1], 'k-', lw=2)
    ax.set_aspect('equal')
    return p1, p2


def plot_est(ax, result):
    for i, (traj_est, landmark_est) in enumerate(result):
        if i == 0:
            p1 = ax.scatter(landmark_est[:, 0], landmark_est[:, 1], marker='+', lw=2, alpha=0.5)
            alpha = 1.0
        else:
            alpha = 0.1
        (p2, ) = ax.plot(traj_est[:, 0], traj_est[:, 1], 'g-', lw=2, alpha=alpha)
    ax.set_aspect('equal')
    return p1, p2


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Design a simulator
    sim = Simulator()
    sim.random_map(50, (-10, 10, -10, 10))
    pose = gtsam.Pose2(5, -5, np.pi / 2.0)
    for i in range(50):
        sim.traj.append(pose)
        if i % 10 == 0 and i > 0:
            u = 0.0, 1.0, np.pi / 2.0
        else:
            u = 1.0, 0.0, 0.0
        pose = pose.compose(gtsam.Pose2(*u))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, True, True, figsize=(8, 3))

    # Perfect DA
    sim.reset()
    result = slam(sim, 'perfect')
    plot_sim(ax1, sim)
    plot_est(ax1, result)
    ax1.set_title('Perfect DA')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)

    # # JCBB
    sim.reset()
    result = slam(sim, 'jcbb')
    p1, p2 = plot_sim(ax2, sim)
    p3, p4 = plot_est(ax2, result)
    ax2.set_title('JCBB')
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)

    # MHJCBB
    sim.reset()
    result = mhjcbb(sim)
    plot_sim(ax3, sim)
    plot_est(ax3, result)
    ax3.set_title('MHJCBB')
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)

    fig.legend((p1, p2, p3, p4), ('Landmarks - True', 'Trajectory - True', 'Landmarks - SLAM', 'Trajectory - SLAM'),
            'lower center', ncol=4, fontsize='medium')

    plt.tight_layout()
    plt.show()