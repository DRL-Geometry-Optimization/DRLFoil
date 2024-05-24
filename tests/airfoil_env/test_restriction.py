import sys

sys.path.append('../drlfoil/')

from drlfoil import BoxRestriction

import pytest

def test_box_init():
    # Test the initialization of the BoxRestriction class
    box = BoxRestriction(0.6, 0.1, 0.3, 0.1)
    assert round(box.posx, 1) == 0.6
    assert round(box.posy, 1) == 0.1
    assert round(box.width, 1) == 0.3
    assert round(box.height, 2) == 0.1
    assert round(box.x1, 2) == 0.45
    assert round(box.x2, 2) == 0.75
    assert round(box.y1, 2) == 0.05
    assert round(box.y2, 2) == 0.15

def test_box_init_out():
    # Test the initialization of the BoxRestriction class outside the environment
    xmin = BoxRestriction._XMIN
    xmax = BoxRestriction._XMAX
    ymin = BoxRestriction._YMIN
    ymax = BoxRestriction._YMAX

    width = 0.3
    height = 0.1
    with pytest.raises(ValueError):
        BoxRestriction(xmin+(width/2)-0.02, 0.1, width, 0.1)

    with pytest.raises(ValueError):
        BoxRestriction(xmax-(width/2)+0.02, 0.1, width, 0.1)

    with pytest.raises(ValueError):
        BoxRestriction(0.6, ymin+(height/2)-0.02, 0.3, height)

    with pytest.raises(ValueError):
        BoxRestriction(0.6, ymax-(height/2)+0.02, 0.3, height)


@pytest.fixture
def box():
    return BoxRestriction(0.5, 0.0, 0.4, 0.2)

def test_box_coordinates(box):
    # Test the coordinates method
    assert box.coordinates() == ((0.3, 0.1), (0.7, 0.1), (0.3, -0.1), (0.7, -0.1))


@pytest.mark.parametrize("x, y, expected", [
    (0.5, 0.0, True),  
    (0.7, 0.1, True),  
    (0.75, 0.0, False),  
    (0.5, 0.2, False),  
    (0.2, 0.27, False),  
])
def test_is_inside(box, x, y, expected):
    assert box.is_inside(x, y) == expected