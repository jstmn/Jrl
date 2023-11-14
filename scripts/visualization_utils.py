from klampt import vis
from klampt.model import coordinates, trajectory

from jrl.robot import Robot


def _init_klampt_vis(robot: Robot, window_title: str, show_collision_capsules: bool = True):
    vis.init()

    background_color = (1, 1, 1, 0.7)
    vis.setBackgroundColor(background_color[0], background_color[1], background_color[2], background_color[3])
    size = 5
    for x0 in range(-size, size + 1):
        for y0 in range(-size, size + 1):
            vis.add(
                f"floor_{x0}_{y0}",
                trajectory.Trajectory([1, 0], [(-size, y0, 0), (size, y0, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )
            vis.add(
                f"floor_{x0}_{y0}2",
                trajectory.Trajectory([1, 0], [(x0, -size, 0), (x0, size, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )

    vis.add("world", robot.klampt_world_model)
    vis.add("coordinates", coordinates.manager())
    vis.setWindowTitle(window_title)
    vis.show()
