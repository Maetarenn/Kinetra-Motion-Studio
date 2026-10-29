import math
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kinetra.ui.stage import StageViewport, StuntProp
from kinetra.ui.main_window import MainWindow
from kinetra.core.body_solver import BodySolver
from kinetra.core.anatomy import limits_for


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_stage_props_spawn_and_bounding_selection(app):
    stage = StageViewport()
    stage.resize(800, 600)
    stage.show()
    QtWidgets.QApplication.processEvents()

    assert len(stage.props) == 0
    box = stage.spawn_prop("box")
    assert len(stage.props) == 1
    assert box.kind == "box"
    assert box.selected is True
    assert stage.selected_prop is box

    # Test gizmo drawing and bounds projection
    stage.repaint()
    QtWidgets.QApplication.processEvents()
    bounds = stage._prop_screen_bounds(box)
    assert bounds is not None
    assert bounds.width() > 0 and bounds.height() > 0

    # Click inside bounds
    center_pt = bounds.center()
    clicked = stage._find_clicked_prop(center_pt)
    assert clicked is box

    # Clear
    stage.clear_props()
    assert len(stage.props) == 0
    assert stage.selected_prop is None
    stage.close()


def test_stage_maximize_toggle(app, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "test.ini"), QtCore.QSettings.IniFormat)
    win = MainWindow(settings)
    win.show()

    assert not getattr(win, "_stage_maximized", False)
    # Trigger maximize
    win.toggle_maximize_stage()
    assert win._stage_maximized is True
    assert win.stage.btn_maximize.is_maximized is True
    # Trigger restore
    win.toggle_maximize_stage()
    assert win._stage_maximized is False
    assert win.stage.btn_maximize.is_maximized is False
    win.close()


def test_multi_actor_toggle(app, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "test.ini"), QtCore.QSettings.IniFormat)
    win = MainWindow(settings)

    assert win.stage.multi_actor is False
    win.toggle_multi_actor(True)
    assert win.stage.multi_actor is True
    assert win.act_multi_actor.isChecked() is True
    win.toggle_multi_actor(False)
    assert win.stage.multi_actor is False
    win.close()


def test_head_touch_reach_flexion_limits():
    solver = BodySolver()
    assert math.isclose(solver.MAX_FLEX, math.radians(160.0), abs_tol=1e-3)
    forearm_limit = limits_for("forearm_L")
    assert forearm_limit[1] >= 160
