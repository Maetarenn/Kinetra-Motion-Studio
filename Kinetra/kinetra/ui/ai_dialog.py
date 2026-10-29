"""AI trainer dialog: build the datasets (synthetic + your takes + BVH libraries) and train the hand and body priors."""
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..ai.trainer import train_all, USER_DIR
from ..i18n import t


class _TrainThread(QtCore.QThread):
    progress = QtCore.Signal(float, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, kw, parent=None):
        super().__init__(parent)
        self.kw = kw

    def run(self):
        try:
            self.done.emit(train_all(progress=lambda f, s: self.progress.emit(f, s), **self.kw))
        except Exception as exc:
            self.failed.emit(str(exc))


class AITrainerDialog(QtWidgets.QDialog):
    trained = QtCore.Signal()

    def __init__(self, project_dir=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("ai.title"))
        self.resize(640, 480)
        self.project_dir, self.bvh_dir, self.thread = project_dir, None, None
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(t("ai.intro"))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QtWidgets.QFormLayout()
        self.chk_takes = QtWidgets.QCheckBox(t("ai.takes"))
        self.chk_takes.setChecked(bool(project_dir))
        self.chk_takes.setEnabled(bool(project_dir))
        self.btn_bvh = QtWidgets.QPushButton(t("ai.bvh"))
        self.btn_bvh.clicked.connect(self._pick_bvh)
        self.lbl_bvh = QtWidgets.QLabel("-")
        self.cb_size = QtWidgets.QComboBox()
        self.cb_size.addItems([t("ai.quick"), t("ai.normal"), t("ai.long")])
        self.cb_size.setCurrentIndex(1)
        form.addRow(self.chk_takes)
        form.addRow(self.btn_bvh, self.lbl_bvh)
        form.addRow(QtWidgets.QLabel(t("ai.epochs")), self.cb_size)
        lay.addLayout(form)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)
        info = USER_DIR / "info.json"
        self.log.appendPlainText(t("ai.status_user") if info.exists() else t("ai.status_default"))
        row = QtWidgets.QHBoxLayout()
        self.btn_train = QtWidgets.QPushButton(t("ai.train"))
        self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self._train)
        btn_close = QtWidgets.QPushButton(t("btn.close"))
        btn_close.clicked.connect(self.accept)
        row.addWidget(self.btn_train)
        row.addStretch(1)
        row.addWidget(btn_close)
        lay.addLayout(row)

    def _pick_bvh(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, t("ai.bvh"))
        if d:
            self.bvh_dir = d
            self.lbl_bvh.setText(d)

    def _train(self):
        n = {0: (80, 8000, 8, 12000), 1: (250, 30000, 20, 40000), 2: (500, 80000, 40, 60000)}[self.cb_size.currentIndex()]
        kw = dict(n_body_seq=n[0], n_windows=n[1], epochs=n[2], n_hand=n[3], bvh_dir=self.bvh_dir,
                  project_dir=self.project_dir if self.chk_takes.isChecked() else None)
        self.btn_train.setEnabled(False)
        self.log.appendPlainText(t("ai.running"))
        self.thread = _TrainThread(kw)
        self.thread.progress.connect(lambda f, s: (self.bar.setValue(int(f * 1000)), None))
        self.thread.done.connect(self._done)
        self.thread.failed.connect(lambda m: (self.log.appendPlainText(t("ai.fail", m)), self.btn_train.setEnabled(True)))
        self.thread.start()

    def _done(self, info):
        self.bar.setValue(1000)
        self.log.appendPlainText(t("ai.done", info["hand_samples"], info["body_sequences"]))
        self.btn_train.setEnabled(True)
        self.trained.emit()

    def done(self, r):
        if self.thread is not None and self.thread.isRunning():
            self.thread.wait(100)
        super().done(r)
