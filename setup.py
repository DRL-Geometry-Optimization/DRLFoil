from setuptools import setup, find_packages

setup(
    name='DRLFoil',
    version='0.1',
    packages=find_packages(),
    description='A Deep Reinforcement Learning Framework for Airfoil Design Optimization.',
    author='Pablo Magarinos Docampo',
    author_email='pablo.magarinos@outlook.com',
    license='MIT',
    install_requires=[
        'torch==2.3.0+cu121',
        'stable-baselines3[extra]==2.3.1',
        'AeroSandbox==4.2.2',
        'gymnasium==0.29.1',
        'NeuralFoil==0.1.10',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)