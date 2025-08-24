from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='dinov3_ros',
    version='0.1.0',
    packages=find_packages(include=['dinov3_toolkit', 'dinov3_toolkit.*']),
    install_requires=parse_requirements('requirements.txt'),
    # package_data={
    #     "dinov3_toolkit": [
    #         "heads/*/weights/*.pth",
    #         "backbones/weights/*.pth"
    #     ]
    # },
    include_package_data=True,
)