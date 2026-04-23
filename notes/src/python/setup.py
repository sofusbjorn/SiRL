from setuptools import find_packages
from setuptools import setup

setup(
    name="sirl-python",
    version="0.0.1",
    description="Python Code Library for Simulation-based Reinforcement Learning Course Notes",
    long_description=open("README.md").read(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "mujoco",
        "mujoco-mjx",
        "pythreejs",
        "gymnasium==0.28.1",
        "stable-baselines3[extra]",
    ],

    keywords="reinforcement learning rigidbody dynamics optimization",
)
