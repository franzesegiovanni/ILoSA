# ILoSA: Interactive Learning of Stiffness and Attractors
Giovanni Franzese<sup>∗</sup>, Anna Mészáros, Luka Peternel, and Jens Kober

**Presented** in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) in Prague, Czech Republic 

**Winner** of BEST LATE BREAKING RESULTS POSTER AWARD in 2021 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM 2021)

**Full paper** is publically available at https://arxiv.org/abs/2103.03099

## What is ILoSA? 

Teaching robots how to apply forces according to our preferences is still an open challenge that has to be tackled from multiple engineering perspectives.
ILoSA is a framework which enables robots to learn variable impedance policies where both the Cartesian stiffness and the attractor can be learned from human demonstrations and corrections with a user-friendly interface. These policies are learned with the help of Gaussian Processes (GPs), exploiting the properties of GPs to identify regions of uncertainty and allow for interactive corrections, stiffness modulation and active disturbance rejection.

The learning process consists of two steps:
1. Kinesthetic demonstration - in which the human leads the robot through the desired motion, and
2. Correction phase - in which the human can locally adapt the initially demonstrated policy so as to improve performance during task execution.

ILoSA has shown to be effective in different challenging tasks, such as pushing a box, board wiping, as well as plugging and unplugging a plug from a socket. Good task execution could be achieved with a single demonstration followed by a short period of corrections. The low time demand coupled with an intuitive correction interface further makes ILoSA accessible to people who may not be experts in robotics. The following [video](https://www.youtube.com/watch?v=MAG-kFGztws) provides an overview of ILoSA's capabilities. 

With this we invite you all to try ILoSA out for yourselves, to expand it to other tasks challenging not only for robots but maybe even humans as well and keep pushing the capabilities of robot learning further.

![Cleaning task](.gif/cleaning_bot.gif)

# How to run the code on a Franka Emika Panda
To install:
- Create a workspace containing a src directory.
- Inside the src directory, clone the franka_ros repository by frankaemika.
```git clone https://github.com/frankaemika/franka_ros ```
- Inside the repository, clone the human friendly controllers of TU Delft https:```//github.com/franzesegiovanni/franka_human_friendly_controllers```
- Return to the workspace main directory (cd ../..).
- Source your version of ROS (e.g. ```source /opt/ros/melodic/setup.bash```).
- Build the project, calling: ```catkin_make -DMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/path/to/libfranka/build``` (be sure that libfranka is installed https://frankaemika.github.io/docs/installation_linux.html)

To run ILoSA:
- Switch on your Panda robot (make sure the gripper is initialized correctly), unlock its joints, and activate the FCI if necessary.
- Open a terminal and be sure that the ros of the new catkin workspace is sourced, i.e. ```source devel/setup.bash```
- ```roslaunch franka_human_friendly_controllers cartesian_variable_impedance_controller.launch robot_ip:=ROBOT_IP load_gripper:=True```
- ``` python3 main.py```

# Cite ILoSA
If you found ILoSA useful for your research, please cite it as:

```
@inproceedings{franzese2021ilosa,
  title={ILoSA: Interactive learning of stiffness and attractors},
  author={Franzese, Giovanni and M{\'e}sz{\'a}ros, Anna and Peternel, Luka and Kober, Jens},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={7778--7785},
  year={2021},
  organization={IEEE}
}
```

# Acknowledgements
The research surrounding this code is funded by the European Research Council Starting Grant TERI “Teaching Robots Interactively”.
