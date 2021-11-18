from setuptools import setup, find_packages
setup(
    name='multiview_calib',
    version='0.1', 
    packages=find_packages(include=['multiview_calib*']),
    url='https://github.com/cvlab-epfl/multiview_calib',
    author='cvlab-epfl',
    install_requires=['numpy', 'scipy', 'imageio', 'matplotlib']
)

