from jkinpylib.evaluation import pose_errors_cm_deg
from jkinpylib.robots import Panda, Fetch, FetchArm
from jkinpylib.utils import set_seed
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# from matplotlib.patches import Rectangle
# from matplotlib.path import Path

np.set_printoptions(suppress=True, linewidth=120)

set_seed()


def sample_from_convex_hull(points: np.ndarray, n: int, ndof: int) -> np.ndarray:
    samples = []
    hull = ConvexHull(points)
    # hull_path = Path( hull.points[hull.vertices])
    print("hull.min_bound:", hull.min_bound)
    print("hull.max_bound:", hull.max_bound)
    # print(hull_path)

    # Define the bounding box of the convex hull
    min_coords = np.min(hull.points, axis=0)
    max_coords = np.max(hull.points, axis=0)

    tri = Delaunay(hull.points[hull.vertices])

    # Generate random points within the bounding box
    random_points = np.random.uniform(low=min_coords, high=max_coords, size=(15, hull.ndim))

    # Check if the random points are inside the convex hull
    inside_hull = tri.find_simplex(random_points) >= 0

    print(inside_hull)
    exit()

    while True:
        rand_points = np.zeros((n, ndof))
        for i in range(ndof):
            rand_points[:, i] = np.random.uniform(hull.min_bound[i], hull.max_bound[i], n)

        rand_points[i] = np.array(
            [np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])]
        )
        # We check if the random point is inside the convex hull, otherwise we draw it again
        while hull_path.contains_point(rand_points[i]) == False:
            rand_points[i] = np.array(
                [np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])]
            )

    # rand_points = np.empty((n, 2))
    # for i in range(n):
    #     #Draw a random point in the bounding box of the convex hull
    #     rand_points[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])
    #     #We check if the random point is inside the convex hull, otherwise we draw it again
    #     while hull_path.contains_point(rand_points[i]) == False:
    #         rand_points[i] = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1])])


""" python scripts/calculate_rotational_repeatability.py
"""

if __name__ == "__main__":
    robot = Panda()
    print(robot.positional_repeatability_mm)

    n_poses = 5
    n_perturbs = 500
    joint_angles, poses = robot.sample_joint_angles_and_poses(n_poses)

    for q, pose in zip(joint_angles, poses):
        print("\nNew pose")

        q_deltas = []

        for _ in range(n_perturbs):
            pertubation = np.random.random(robot.n_dofs) - 0.5
            pertubation = 0.00001 * pertubation / np.linalg.norm(pertubation)
            q_perturbed = q.copy()
            while True:
                q_perturbed_last = q_perturbed
                q_perturbed += pertubation
                pos_error_cm, rot_error_deg = pose_errors_cm_deg(
                    robot.forward_kinematics_klampt(q_perturbed[None, :]), pose[None, :], acos_epsilon=1e-30
                )
                pos_error_mm = pos_error_cm[0] * 10

                if pos_error_mm > robot.positional_repeatability_mm:
                    q_perturbed = q_perturbed_last
                    break

            joint_angle_diff = q_perturbed - q
            q_deltas.append(joint_angle_diff)

        # TODO: Create a convex hull and sample from it. https://stackoverflow.com/a/67178146/5191069
        q_deltas = np.array(q_deltas)

        n = 10
        sample_from_convex_hull(q_deltas, n, robot.n_dofs)

        # Draw n random points inside the convex hull

        # q_deltas = np.array(q_deltas)
        # pos = q_deltas[:, 0:2]
        # hull = ConvexHull( pos )
        # #Bounding box
        # bbox = [hull.min_bound, hull.max_bound]
        # hull_path = Path( hull.points[hull.vertices])
        # #Draw n random points inside the convex hull
        # n = 1000

        # Plot
        # plt.figure(figsize=(10, 10))
        # plt.scatter(pos[:, 0], pos[:, 1], marker='o',  c='blue', alpha = 1, label ='Initial points')
        # for simplex in hull.simplices:
        #         plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], '-k')
        # plt.gca().add_patch(Rectangle((bbox[0][0], bbox[0][1]), bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1],facecolor = 'None', edgecolor = 'cyan'))
        # plt.scatter(rand_points[:, 0], rand_points[:, 1], marker='o',  c='red', alpha = 0.31, label ='Random points inside hull')
        # plt.legend()
        # plt.title("Convex hull of joint angle pertubations that nearly surpass the robot's \n positional mechanical repeatability, as well as points sample from inside the hull")
        # plt.xlabel(r"\theta_1")
        # plt.ylabel(r"\theta_2")
        # plt.show()
        # plt.close()
