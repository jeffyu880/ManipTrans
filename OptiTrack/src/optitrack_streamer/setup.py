import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'optitrack_streamer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'calibration'), glob('*.calib.json'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kaijunge',
    maintainer_email='kai.junge@epfl.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'optitrack_streamer_node = optitrack_streamer.optitrack_streamer_node:main',
            'wrist_streamer_node = optitrack_streamer.wrist_streamer_node:main'
        ],
    },
)
