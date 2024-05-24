import sys

sys.path.append('../drlfoil/')

import numpy as np
import random 
import pytest
from drlfoil.airfoil_env import AirfoilTools
from drlfoil import BoxRestriction


def test_kulfan():
    """
    Test the Kulfan airfoil generation. 
    The test checks if the weights are correctly assigned to the airfoil object.
    """
    airfoil = AirfoilTools()
    upper_weights = 0.1*np.ones(10)
    lower_weights = -0.1*np.ones(10)
    leading_edge_weight = 0.1
    TE_thickness = 0.0
    name = "Test Airfoil"

    airfoil.kulfan(upper_weights, lower_weights, leading_edge_weight, TE_thickness, name)
    
    # Check if the weights are correctly assigned
    assert np.array_equal(airfoil.upper_weights, upper_weights)
    assert np.array_equal(airfoil.lower_weights, lower_weights)
    assert airfoil.airfoil.leading_edge_weight == leading_edge_weight
    assert airfoil.airfoil.TE_thickness == TE_thickness

    # Check if the length of the weights is correct
    assert len(airfoil.lower_weights) == airfoil.numparams
    assert len(airfoil.upper_weights) == airfoil.numparams


@pytest.fixture
def airfoil():
    """
    Fixture that returns a Kulfan airfoil object with default parameters
    """
    foil = AirfoilTools()
    upper_weights = 0.1*np.ones(10)
    lower_weights = -0.1*np.ones(10)
    leading_edge_weight = 0.1
    TE_thickness = 0.0
    foil.kulfan(upper_weights, lower_weights, leading_edge_weight, TE_thickness)
    return foil
    

def test_random_kulfan(airfoil):
    """
    Test the random_kulfan method.
    """
    airfoil.random_kulfan()
    # Check if the length of the weights is correct
    assert len(airfoil.airfoil.upper_weights) == airfoil.numparams
    assert len(airfoil.airfoil.lower_weights) == airfoil.numparams


def test_random_kulfan2(airfoil):
    """
    Test the random_kulfan2 method.
    """
    airfoil.random_kulfan2()
    
    assert len(airfoil.airfoil.upper_weights) == airfoil.numparams
    assert len(airfoil.airfoil.lower_weights) == airfoil.numparams


def test_modify_airfoil(airfoil):
    """
    Test the modify_airfoil method. 
    The test checks if the weights are correctly modified.
    """
    # Save the previous weights
    previous_upper = airfoil.airfoil.upper_weights.copy()
    previous_lower = airfoil.airfoil.lower_weights.copy()
    previous_leading_edge = airfoil.airfoil.leading_edge_weight

    # Generate a random action
    action=np.random.rand(airfoil.numparams*2+1)

    # Modify the airfoil
    name = "Modified Airfoil"
    airfoil.modify_airfoil(action, name=name)

    # Check if the weights are correctly modified
    assert np.array_equal(airfoil.airfoil.upper_weights, previous_upper + action[:airfoil.numparams])
    assert np.array_equal(airfoil.airfoil.lower_weights, previous_lower + action[airfoil.numparams: -1])
    assert airfoil.airfoil.leading_edge_weight == previous_leading_edge + action[-1]
    assert airfoil.airfoil.TE_thickness == 0.0
    assert airfoil.airfoil.name == name


def test_modify_airfoil_unit_up(airfoil):
    """
    Test the modify_airfoil_unit method.
    The test checks if the weights are correctly modified.
    Testing upper surface.
    """
    previous_upper = airfoil.airfoil.upper_weights.copy()
    previous_lower = airfoil.airfoil.lower_weights.copy()
    previous_leading_edge = airfoil.airfoil.leading_edge_weight

    face = "up"
    index = 1
    variation = 0.2

    airfoil.modify_airfoil_unit(face, index, variation)

    # Modify the weights of the upper surface
    previous_upper[index] += variation
    # Check if the weights are the same
    assert np.array_equal(airfoil.airfoil.upper_weights, previous_upper)
    assert np.array_equal(airfoil.airfoil.lower_weights, previous_lower)
    assert airfoil.airfoil.leading_edge_weight == previous_leading_edge
    assert airfoil.airfoil.TE_thickness == 0.0


def test_modify_airfoil_unit_down(airfoil):
    """
    Test the modify_airfoil_unit method.
    The test checks if the weights are correctly modified.
    Testing lower surface.
    """
    previous_upper = airfoil.airfoil.upper_weights.copy()
    previous_lower = airfoil.airfoil.lower_weights.copy()
    previous_leading_edge = airfoil.airfoil.leading_edge_weight

    face = "down"
    index = 1
    variation = 0.2

    airfoil.modify_airfoil_unit(face, index, variation)

    previous_lower[index] += variation

    assert np.array_equal(airfoil.airfoil.upper_weights, previous_upper)
    assert np.array_equal(airfoil.airfoil.lower_weights, previous_lower)
    assert airfoil.airfoil.leading_edge_weight == previous_leading_edge
    assert airfoil.airfoil.TE_thickness == 0.0



def test_analysis(airfoil):
    """
    The test checks if the aerodynamics are calculated
    """
    angle = 0.0
    re = 1e6
    model = "xlarge"
    airfoil.analysis(angle, re, model)
    assert airfoil.aerodynamics is not None
    assert airfoil.aerodynamics["CL"] is not None
    assert airfoil.aerodynamics["CD"] is not None


def test_get_cl(airfoil):
    """
    Tests if the get_cl method returns the correct value
    """
    airfoil.analysis(0.0, 1e6, "xlarge")
    assert airfoil.get_cl() == airfoil.aerodynamics["CL"][0]


def test_get_cd(airfoil):
    """
    Tests if the get_cd method returns the correct value
    """
    airfoil.analysis(0.0, 1e6, "xlarge")
    assert airfoil.get_cd() == airfoil.aerodynamics["CD"][0]


def test_get_efficiency(airfoil):
    """
    Tests if the get_efficiency method returns the correct value
    """
    airfoil.analysis(0.0, 1e6, "xlarge")
    assert airfoil.get_efficiency() == airfoil.aerodynamics["CL"][0] / airfoil.aerodynamics["CD"][0]


def test_check_airfoil(airfoil):
    """
    Test the check_airfoil method.
    The test checks if the default airfoil is valid.
    """
    assert airfoil.check_airfoil() == True

def test_check_airfoil_checksurfaces():
    """
    Test if an airfoil with inverted curves is invalid.
    """
    airfoil = AirfoilTools()
    # The curves are inverted so the airfoil should be invalid
    upper_weights = -0.1*np.ones(10) 
    lower_weights = 0.1*np.ones(10)
    leading_edge_weight = 0.1
    TE_thickness = 0.0
    airfoil.kulfan(upper_weights, lower_weights, leading_edge_weight, TE_thickness)
    assert airfoil.check_airfoil() == False

 
def test_check_airfoil_checksurfaces2():
    """
    Test if an airfoil with an intersected curve is invalid.
    """

    airfoil = AirfoilTools()
    # Here, the airfoil is valid
    upper_weights = 0.1*np.ones(10) 
    lower_weights = -0.1*np.ones(10)
    leading_edge_weight = 0.1
    TE_thickness = 0.0
    # Now, we break the airfoil by changing one weight of the lower surface. At this point, the airfoil should be invalid
    lower_weights[3] = 0.7
    airfoil.kulfan(upper_weights, lower_weights, leading_edge_weight, TE_thickness)
    assert airfoil.check_airfoil() == False


def test_get_boxes(airfoil):
    """
    Test the get_boxes method. 
    The test checks if the boxes are correctly assigned to the airfoil object.
    """
    box = BoxRestriction(0.5, 0, 0.3, 0.05)
    airfoil.get_boxes(box)
    assert airfoil.boxes == [box]

def test_return_boxes(airfoil):
    """
    Test the return_boxes method.
    The test checks if the boxes are correctly returned.
    """
    box = BoxRestriction(0.5, 0, 0.3, 0.05)
    airfoil.get_boxes(box)
    assert airfoil.return_boxes() == [[0.5, 0, 0.3, 0.05]]
