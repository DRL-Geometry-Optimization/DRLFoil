import sys

sys.path.append('../drlfoil/')

import numpy as np
import pytest
from drlfoil import Optimize

@pytest.fixture
def optimize():
    model = "nobox"
    cl_target = 0.5
    reynolds = 1e6
    boxes=[]
    steps = 10
    logs = 1

    optimize = Optimize(model = model, 
                        cl_target=cl_target, 
                        reynolds=reynolds, 
                        boxes=boxes,
                        steps=steps, 
                        logs = logs)
    return optimize

def test_init(optimize):
    #assert optimize.model is None
    #assert optimize.env is None
    assert optimize.cl_target == 0.5
    assert optimize.reynolds == 1e6
    assert optimize.steps == 10
    assert optimize.bestairfoil is None
    assert optimize.bestairfoil_reward == -float("inf")
    assert optimize.boxes == []

"""
def test_run(optimize):
    optimize.run()
    assert optimize.bestairfoil is not None
    assert optimize.bestairfoil_reward > -float("inf")

def test_save(optimize):
    optimize.run()
    optimize.save("best_airfoil")
    # TODO: Add assertions to check if the file was saved correctly

def test_reset(optimize):
    optimize.reset(reynolds=200000, cl_target=0.6)
    assert optimize.reynolds == 200000
    assert optimize.cl_target == 0.6
    assert optimize.bestairfoil is None

def test_show(optimize):
    optimize.run()
    optimize.show()
    # TODO: Add assertions to check if the airfoil plot is displayed correctly

def test_analyze(optimize):
    optimize.run()
    analysis = optimize.analyze(plot=False)
    assert isinstance(analysis, dict)
    assert "lift_coefficient" in analysis
    assert "drag_coefficient" in analysis
    assert "efficiency" in analysis
"""