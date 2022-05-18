# ILoSA: Interactive Learning of Stiffness and Attractors
Giovanni Franzese∗, Anna Mészáros, Luka Peternel, and Jens Kober

**Presented** in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) in Prague, Czech Republic 

**Winner** of BEST LATE BREAKING RESULTS POSTER AWARD in 2021 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM 2021)

## What is ILoSA? 

# How to run the code on a Franka Emika Panda
To install:
- Create a workspace containing a src directory.
- Inside the src directory, clone the franka_ros repository by frankaemika. https://github.com/frankaemika/franka_ros
- Inside the repository, clone the human friendly controllers of TU Delft https://github.com/franzesegiovanni/franka_human_friendly_controllers
- Return to the workspace main directory (cd ../..).
- Source your version of ROS (e.g. source /opt/ros/melodic/setup.bash).
- Build the project, calling: catkin_make -DMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=/path/to/libfranka/build (be sure that libfranka is installed https://frankaemika.github.io/docs/installation_linux.html)

To run ILoSA:
- Switch on your Panda robot (make sure the gripper is initialized correctly), unlock its joints (and activate the FCI if necessary).
- Open a terminal and be sure that the ros of the new catkin workspace is sourced, i.e. source devel/setup.bash
- roslaunch franka_human_friendly_controllers cartesian_variable_impedance_controller.launch robot_ip:=ROBOT_IP load_gripper:=True
- python3 main.py
