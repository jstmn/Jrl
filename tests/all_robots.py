from jrl.robots import get_all_robots

all_robots = get_all_robots()
for robot in all_robots:
    print(f"Loaded robot: {robot}")
