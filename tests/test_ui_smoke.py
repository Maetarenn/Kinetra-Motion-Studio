import time
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from kinetra import i18n
from kinetra.capture.pipeline import CaptureWorker, CaptureConfig
from test_pipeline import SynthSource, FakeModels
from test_body_solver import synth_pose


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)


def test_window_runs_switches_languages_and_records(app, tmp_path, monkeypatch):
    from kinetra.ui import main_window as mw
    from kinetra.ui.theme import QSS
    from kinetra.core.project import Project
    app.setStyleSheet(QSS)
    truth = synth_pose() + np.array([0, 0.97, 0])
    orig = mw.CaptureWorker
    monkeypatch.setattr(mw, "CaptureWorker", lambda cfg: orig(
        CaptureConfig(sources=[SynthSource(n=400)], track_face=False), models=FakeModels(lambda: truth)))
    settings = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)
    win = mw.MainWindow(settings)
    win.show()
    win._set_project(Project.create(tmp_path / "proj", "Smoke"))
    win.start_capture()
    pump(2500)
    assert win._latest is not None and win._latest["joints"] is not None
    win.btn_rec.setChecked(True)
    pump(1500)
    for code in i18n.available():                       # live language switch must never fail
        win.set_language(code)
        pump(30)
    win.set_language("tr")
    assert win.btn_start.text() == "BAŞLAT"
    win.tabs.setCurrentIndex(1); pump(200)
    win.tabs.setCurrentIndex(2); pump(100)
    win.btn_rec.setChecked(False)
    assert len(win.project.takes()) == 1
    win.stop_capture()
    pump(300)
    assert not win.worker.isRunning()
    win.close()
