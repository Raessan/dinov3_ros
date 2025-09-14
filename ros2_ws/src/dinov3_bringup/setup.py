from setuptools import find_packages, setup
import glob
import os

package_name = 'dinov3_bringup'

# Collect all launch files in the launch folder
launch_files = glob.glob(os.path.join('launch', '*.py'))
config_files =  glob.glob(os.path.join('config', '*.yaml'))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', launch_files),
        ('share/' + package_name + '/config', config_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='raessan@outlook.com',
    description='Package to launch dinov3 node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dinov3_node = dinov3_ros.dinov3_node:main',
        ],
    },
)
