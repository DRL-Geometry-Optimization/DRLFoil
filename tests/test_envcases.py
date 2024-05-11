import sys

sys.path.append('../drlfoil/')

import numpy as np
import pytest
from drlfoil.airfoil_env import AirfoilEnv




def test_airfoil_env_obs():
    # Test case 1: 0 boxes, fixed Reynolds, cl_reset None
    env = AirfoilEnv(n_boxes=0, reynolds=1e6, cl_reset=None, max_steps=2, render_mode="no_display",
                     airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0])
    env.reset()
    observation, _, _, _, _ = env.step(np.concatenate([0.5*np.ones(10), 0.5*np.ones(10), [0.0]]))
    print(len(observation["airfoil"]))
    assert isinstance(observation, dict)
    assert len(observation["airfoil"]) == 10*2+1
    assert len(observation["reynolds"]) == 1
    assert len(observation["cl_target"]) == 1
    assert "boxes" not in observation

    

    # Test case 2: 1 box, random Reynolds, cl_reset 0.5
    env = AirfoilEnv(n_boxes=1, reynolds=None, cl_reset=0.5, max_steps=2, render_mode="no_display",
                     airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0])
    env.reset()
    observation, _, _, _, _ = env.step(np.concatenate([0.5*np.ones(10), 0.5*np.ones(10), [0.0]]))
    assert isinstance(observation, dict)
    assert len(observation["airfoil"]) == 10*2+1
    assert len(observation["reynolds"]) == 1
    assert len(observation["cl_target"]) == 1
    assert len(observation["boxes"]) == 4*1

    

    # Test case 3: 2 boxes, disabled Reynolds, cl_reset 0.5
    env = AirfoilEnv(n_boxes=2, reynolds=-1, cl_reset=0.5, max_steps=2, render_mode="no_display",
                     airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0])
    env.reset()
    observation, _, _, _, _ = env.step(np.concatenate([0.5*np.ones(10), 0.5*np.ones(10), [0.0]]))
    assert isinstance(observation, dict)
    assert len(observation["airfoil"]) == 10*2+1
    assert "reynolds" not in observation
    assert len(observation["cl_target"]) == 1
    assert len(observation["boxes"]) == 4*2


    # Test case 4: 0 boxes, disabled Reynolds, cl_reset None
    env = AirfoilEnv(n_boxes=0, reynolds=-1, cl_reset=None, max_steps=2, render_mode="no_display",
                     airfoil_seed=[0.1*np.ones(10), -0.1*np.ones(10), 0.0])
    env.reset()
    observation, _, _, _, _ = env.step(np.concatenate([0.5*np.ones(10), 0.5*np.ones(10), [0.0]]))
    assert isinstance(observation, dict)
    assert len(observation["airfoil"]) == 10*2+1
    assert "reynolds" not in observation
    assert len(observation["cl_target"]) == 1
    assert "boxes" not in observation


def test_limits_reynolds():
    with pytest.raises(Exception):
        env = AirfoilEnv(n_boxes=1, reynolds=1e10, cl_reset=None, max_steps=2, render_mode="no_display",
                         airfoil_seed=[0.1*np.ones(10), -0.1*np.ones(10), 0.0])
        



def test_limits_Cl():
    with pytest.raises(Exception):
        env = AirfoilEnv(n_boxes=1, reynolds=None, cl_reset=2, max_steps=2, render_mode="no_display",
                         airfoil_seed=[0.1*np.ones(10), -0.1*np.ones(10), 0.0])




def test_randomness():
    # Test case: cl_reset and Reynolds are None
    env = AirfoilEnv(n_boxes=1, reynolds=None, cl_reset=None, max_steps=2, render_mode="no_display",
                     airfoil_seed = [0.1*np.ones(10), -0.1*np.ones(10), 0.0])

    env.reset()
    observation1, _, _, _, _ = env.step(np.concatenate([0.5*np.ones(10), 0.5*np.ones(10), [0.0]]))
    cl_reset_value1 = observation1["cl_target"]
    reynolds_value1 = observation1["reynolds"]

    env.reset()
    observation2, _, _, _, _ = env.step(np.concatenate([0.5*np.ones(10), 0.5*np.ones(10), [0.0]]))
    cl_reset_value2 = observation2["cl_target"]
    reynolds_value2 = observation2["reynolds"]

    # Check if the values are different
    assert cl_reset_value1 != cl_reset_value2, "cl_reset values are not random"
    assert reynolds_value1 != reynolds_value2, "Reynolds values are not random"



