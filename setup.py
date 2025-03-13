from setuptools import setup, find_packages

setup(name='boxjump',
      version='0.1',
      description='Box Jump MARL environment with graphics.',
      author='Zak Buzzard',
      email='zakbuzzard1@gmail.com',
      packages=find_packages(),
      install_requires=[
            'swig',
            'box2d-py',
            'numpy',
            'pygame',
            'pettingzoo>=1.22',
            'gymnasium',
      ]
)
