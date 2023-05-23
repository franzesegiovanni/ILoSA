import rospkg
import sys
# Create an instance of the ROS package manager
rospack = rospkg.RosPack()

# Get the path to the desired ROS package
package_path = rospack.get_path('franka_msgs')
# Add the package path to sys.path
sys.path.append(package_path)