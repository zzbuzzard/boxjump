from setuptools import setup

setup(name='box_jump',
      version='1.0',
      description='Box Jump MARL environment with graphics.',
      author='Zak Buzzard',
      email='zakbuzzard1@gmail.com',
      install_requires=[
            'swig',
            'box2d-py',
            'numpy',
            'pygame',
            'pettingzoo>=1.24',
            'gymnasium',
      ]
)
