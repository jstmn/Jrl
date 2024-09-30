import rerun as rr
import numpy as np
from math import tau
from rerun.utilities import build_color_spiral
from rerun.utilities import bounce_lerp
from rerun_loader_urdf import URDFLogger

rr.init("rerun_example_my_data", spawn=True)
rr.set_time_seconds("stable_time", 0)

# rr.stdout() # The most important part of this: log to standard output so the Rerun Viewer can ingest it!

# positions = np.zeros((10, 3))
# positions[:, 0] = np.linspace(-10, 10, 10)
# colors = np.zeros((10, 3), dtype=np.uint8)
# colors[:, 0] = np.linspace(0, 255, 10)
# rr.log("my_points", rr.Points3D(positions, colors=colors, radii=0.5))


# Points and colors are both np.array((NUM_POINTS, 3))
NUM_POINTS = 100
points1, colors1 = build_color_spiral(NUM_POINTS)
points2, colors2 = build_color_spiral(NUM_POINTS, angular_offset=tau*0.5)
offsets = np.random.rand(NUM_POINTS)
beads = [bounce_lerp(points1[n], points2[n], offsets[n]) for n in range(NUM_POINTS)]
colors = [[int(bounce_lerp(80, 230, offsets[n] * 2))] for n in range(NUM_POINTS)]
rr.log(
    "dna/structure/scaffolding/beads",
    rr.Points3D(beads, radii=0.06, colors=np.repeat(colors, 3, axis=-1)),
)
rr.log(
    "dna/structure/scaffolding",
    rr.LineStrips3D(np.stack((points1, points2), axis=1), colors=[128, 128, 128])
)
rr.log("dna/structure/left", rr.Points3D(points1, colors=colors1, radii=0.08))
rr.log("dna/structure/right", rr.Points3D(points2, colors=colors2, radii=0.08))


# prefix = os.path.basename(args.filepath)
urdf_filepath = "/home/jstm/.cache/jrl/temp_urdfs/panda_arm_hand_formatted_link_filepaths_absolute.urdf"
# prefix = "my_prefix"
prefix = None
urdf_logger = URDFLogger(urdf_filepath, prefix)
urdf_logger.log()
print("here")

time_offsets = np.random.rand(NUM_POINTS)

for i in range(400):
    time = i * 0.01
    rr.set_time_seconds("stable_time", time)

    times = np.repeat(time, NUM_POINTS) + time_offsets
    beads = [bounce_lerp(points1[n], points2[n], times[n]) for n in range(NUM_POINTS)]
    colors = [[int(bounce_lerp(80, 230, times[n] * 2))] for n in range(NUM_POINTS)]
    rr.log(
        "dna/structure/scaffolding/beads",
        rr.Points3D(beads, radii=0.06, colors=np.repeat(colors, 3, axis=-1)),
    )
    rr.log(
        "dna/structure",
        rr.Transform3D(rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=time / 4.0 * tau)),
    )

