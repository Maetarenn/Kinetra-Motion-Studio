import time
import numpy as np
import pytest
from PySide6 import QtWidgets
from test_pipeline import SynthSource, FakeModels, Res, L
from test_body_solver import synth_pose


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pump(ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)


def test_face_scan_dialog_opens_runs_and_closes(app, tmp_path):
    from kinetra.ui.scan_dialog import FaceScanDialog
    from kinetra.ui.theme import QSS
    app.setStyleSheet(QSS)

    class NoFace(FakeModels):
        def create_face(self, num_faces=1, prefer_gpu=False, log=print):
            class F:
                def detect_for_video(self, image, ts):
                    return Res(None, None, [])
                def close(self): pass
            return F(), "CPU"

    dlg = FaceScanDialog(0, "640x480 @60", tmp_path / "rig.json", models=NoFace(lambda: None), src=SynthSource(n=60))
    dlg.show()
    pump(800)
    assert dlg.view.frame is not None and dlg.bar.maximum() == 12
    dlg.accept()


def test_ai_trainer_dialog_trains_a_quick_model(app, tmp_path, monkeypatch):
    from kinetra.ui import ai_dialog
    from kinetra.ai import trainer
    monkeypatch.setattr(trainer, "USER_DIR", tmp_path / "ai")
    monkeypatch.setattr(ai_dialog, "USER_DIR", tmp_path / "ai")
    real = trainer.train_all
    monkeypatch.setattr(ai_dialog, "train_all", lambda **kw: real(**{**kw, "n_body_seq": 12, "n_windows": 800, "epochs": 1,
                                                                    "n_hand": 1500, "K": 8, "out_dir": tmp_path / "ai"}))
    dlg = ai_dialog.AITrainerDialog(None)
    dlg.show()
    dlg._train()
    t0 = time.time()
    while dlg.thread.isRunning() and time.time() - t0 < 60:
        pump(100)
    pump(100)
    assert (tmp_path / "ai" / "hand_prior.npz").exists() and (tmp_path / "ai" / "body_prior.npz").exists()
    assert trainer.load_models()[0] is not None
