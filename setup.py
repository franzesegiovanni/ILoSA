from setuptools import setup

setup(
    name='ILoSA',
    version='0.0.2',
    description='Interactive Learning of Stiffness and Attractors',
    author='Giovanni Franzese',
    author_email='g.franzese@tudelft.nl',
    packages=['ILoSA', 'franka_gripper', 'franka_msgs'],
    install_requires=[
        'matplotlib',
        'numpy',
	'pandas',
	'pynput',
	'rospy',
	'scikit_learn',
	'scipy',
	'sensor_msgs'],
    # Add other dependencies here
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
