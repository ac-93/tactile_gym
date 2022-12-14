from setuptools import find_packages, setup
import os
import pathlib


lib_folder = os.path.dirname(os.path.realpath(__file__))

# get required packages from requirements.txt
requirement_path = os.path.join(lib_folder, 'requirements.txt')
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

packages = [package for package in find_packages() if package.startswith("tactile_gym")]

# get req data files
data_files = []
ext_list = '.urdf .sdf .xml .STL .stl .ini .obj .mtl .png .npy'.split()
data_path = os.path.join(lib_folder, "tactile_gym", "assets")
for root, dirs, files in os.walk(data_path):
    for fn in files:
        ext = pathlib.Path(fn).suffix
        if ext and ext in ext_list:
            req_file = os.path.join(root, fn)
            data_files.append(req_file)

setup(name='tactile_gym',
      version='0.0.1',
      description='Pybullet environments targeted towards tactile reinforcement learning',
      author='Alex Church',
      author_email='alexhurch1993@gmail.com',
      license='MIT',
      packages=packages,
      package_data={'tactile_gym': data_files},
      install_requires=install_requires,
      zip_safe=False)
