from setuptools import find_packages, setup

package_name = 'dinov3_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rafael Escarabajal',
    maintainer_email='raessan@outlook.com',
    description='Package that use dinov3 as a backbone and light heads to perform computer vision tasks',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dinov3_node = dinov3_ros.dinov3_node:main'
        ],
    },
)
