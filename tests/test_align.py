import numpy as np
import importlib
import os
import sys

# Ensure the repository root is on the path so that `align` can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
align = importlib.import_module('align')

class DummyPlotter:
    def __init__(self):
        self.cleared = False
        self.mesh_added = None
        self.text = ''
    def clear(self):
        self.cleared = True
    def add_mesh(self, mesh, pickable=True, show_edges=True):
        self.mesh_added = mesh
    def render(self):
        pass
    def add_text(self, text, **kwargs):
        self.text = text

class DummyMesh:
    def __init__(self, points):
        self.points = np.array(points)
    def copy(self):
        return DummyMesh(self.points.copy())
    def save(self, *a, **kw):
        pass

def setup_module(module):
    align.plotter = DummyPlotter()
    align.mesh = DummyMesh([[0,0,0],[1,0,0],[0,1,0]])
    align.mesh_transformed = None
    align.picked_points = []
    align.plane_points = {'first': [], 'second': []}
    align.axes_confirmed = {'first': None, 'second': None}
    align.current_axis_selection = 'first'


def test_reset_selection():
    align.picked_points = [[1,2,3]]
    align.plane_points = {'first': [[0,0,0]], 'second': [[1,1,1]]}
    align.axes_confirmed = {'first': 'x', 'second': 'y'}
    align.current_axis_selection = 'second'
    align.mesh_transformed = DummyMesh([[0,0,0]])
    align.reset_selection(align.plotter)
    assert align.picked_points == []
    assert align.plane_points == {'first': [], 'second': []}
    assert align.axes_confirmed == {'first': None, 'second': None}
    assert align.current_axis_selection == 'first'


def test_align_and_transform_list_handling():
    align.mesh = DummyMesh([[0,0,0],[1,0,0],[0,1,0]])
    align.mesh_transformed = None
    align.plane_points = {
        'first': [[0,0,0],[0,1,0],[0,0,1]],
        'second': [[0,0,0],[1,0,0],[0,1,0]],
    }
    align.axes_confirmed = {'first': 'x', 'second': 'z'}
    align.align_and_transform(align.plotter)
    assert align.plane_points == {'first': [], 'second': []}
    assert align.axes_confirmed == {'first': None, 'second': None}
    assert isinstance(align.mesh_transformed.points, np.ndarray)

