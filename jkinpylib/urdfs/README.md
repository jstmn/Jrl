# URDFS



| Robot         | Url | Notes |
| ------------- | ------------- | ------------- |
| `panda`  | https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/models/assets/bullet_data/panda_description/urdf/panda.urdf  |  |
| `fetch` | https://github.com/openai/roboschool/tree/master/roboschool/models_robot/fetch_description/robots  |  |



**A note about the Panda Arm:**

There are two different joint limit sets for the panda arm floating around. One is from `franka_description` and `robosuite`, the other from `StandfordASL/PandaRobot.jl`. 

1. From http://wiki.ros.org/franka_description 'robots/panda/joint_limits.yaml', also from `robosuite` https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/models/assets/bullet_data/panda_description/

```
joint1: -2.8973 2.8973
joint2: -1.7628 1.7628
joint3: -2.8973 2.8973
joint4: -3.0718 -0.0698
joint5: -2.8973 2.8973
joint6: -0.0175 3.7525
joint7: -2.8973 2.8973
# sum joint range: -33.476
```

2. From https://github.com/StanfordASL/PandaRobot.jl/tree/master/deps/Panda 

The range for every joint is larger than for the frank_description. Note: this is in discussion on this github issue: https://github.com/StanfordASL/PandaRobot.jl/issues/1
```
joint1: -2.9671 2.9671
joint2: -1.8326 1.8326
joint3: -2.9671 2.9671
joint4: -3.1416 0.0
joint5: -2.9671 2.9671
joint6: -0.0873 3.8223
joint7: -2.9671 2.9671
# sum joint range: -34.4532
```