import rerun as rr
import numpy as np
import math
from rerun.utilities import build_color_spiral
from rerun.utilities import bounce_lerp
from rerun_loader_urdf import URDFLogger

rr.init("rerun_example_my_data", spawn=True)
rr.set_time_seconds("stable_time", 0)

# prefix = os.path.basename(args.filepath)
urdf_filepath = "/home/jstm/.cache/jrl/temp_urdfs/panda_arm_hand_formatted_link_filepaths_absolute.urdf"
# prefix = "my_prefix"
prefix = None
urdf_logger = URDFLogger(urdf_filepath, prefix)
urdf_logger.log()
print("here")

TODO: make own urdf logger
TODO: add floor

# for i in range(400):
#     time = i * 0.01
#     rr.set_time_seconds("stable_time", time)

# Log the data on a timeline called "step".
for step in range(0, 64):
    rr.set_time_sequence("step", step)
    rr.log("scalar", rr.Scalar(math.sin(step / 10.0)))