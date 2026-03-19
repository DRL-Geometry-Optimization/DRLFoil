# ***DRLFoil***: A Neural Network Approach to Airfoil Optimization

DRLFoil (*Deep Reinforcement Learning Foil*) is a library developed to provide an optimization framework where you can specify lift coefficient and dimensional constraints. The project began development in the context of a Final Degree Thesis, and it is currently in the alpha stage, so all contributions and feedback are appreciated!

The key features of DRLFoil are:
- ***High-Speed Optimization***: The models can find the optimal airfoil not in seconds or tenths of seconds, but in **milliseconds**.
  
- ***Low Resource Requirements***: As the neural networks used are very small, the models can be operated on any computer and even embedded systems (with appropriate adaptations).
  
- ***High Flexibility with Constraints***: The models are trained to accommodate box constraints, cl targets, and a wide range of Reynolds numbers.

>[!NOTE]
> The project has been archived as I am currently involved in other projects related to AI and Reinforcement Learning, though of a different nature. I would love to continue with DRLFoil and take it into the 3D domain, which was the original goal, but I believe there are already incredible people working on these things, and other challenges excite me more.
> For me, DRLFoil was my introduction to software projects (and it came before AI agents, for which I am very grateful) and it has led me to research what I research today. If anyone feels inspired by this work or would like to know more about any aspect of this project, please do not hesitate to reach out, as nothing would make me happier than seeing DRLFoil contribute something to someone.

## Overview
DRLFoil works with three main libraries:
- **Neuralfoil**: This library works with Physics-Informed Neural Networks trained with XFoil, making the iterations significantly faster than with the aforementioned software.
  
- **Stable Baselines 3**: This is one of the most popular Reinforcement Learning libraries, providing the most important algorithms and tools for developing models.
  
- **Gymnasium**: An OpenAI framework that simplifies the creation of environments for deployment. It is fully compatible with SB3.

As it was developed by an aerospace engineer for other engineers, one of the main objectives was to make it easy to use, removing unnecessary steps and making tools as accessible as possible.
Optimizing an airfoil is as easy as writing a few lines of code:

```python
import drlfoil
from drlfoil import BoxRestriction

optimization = drlfoil.Optimize('onebox', cl = 0.8, reynolds = 1e7, boxes=[BoxRestriction(0.4, 0.0, 0.4, 0.15)])
optimization.run()
```
Examples can be found in the *tutorials* folder.

## Installation
It is highly recommended to use Python 3.10.11 as it was the version used during development. However, subsequent versions of Python should not be incompatible with the libraries used.

If you want to install the requirements, here is the command:
```cmd
pip install git+https://github.com/DRL-Geometry-Optimization/DRLFoil
```

If you want to install requirements, the command is the following:
```cmd
pip install -r requirements.txt
```
