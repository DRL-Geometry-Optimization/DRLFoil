from setuptools import setup, find_packages

setup(
    name='airfoil_optimizer',
    version='0.1',
    packages=find_packages(),
    description='Airfoil optimization tool based on deep reinforcement learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Pablo Magarinos Docampo',
    author_email='pablo.magarinos@outlook.com',
    url='https://github.com/DRL-Geometry-Optimization/2d-geometry-optimization-.git',
    install_requires=[
    'torch==2.3.0+cu121',
    'stable-baselines3[extra]==2.3.0a2',
    'AeroSandbox==4.2.2',
    'gymnasium==0.29.1',
    'NeuralFoil==0.1.10',
    'Tensorboard'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='deep reinforcement learning airfoil optimization',  # Palabras clave relevantes
)