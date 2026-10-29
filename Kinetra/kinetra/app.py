"""Application entry point."""
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from . import APP_NAME, ORG_NAME
from .ui.main_window import MainWindow, ASSETS
from .ui.theme import QSS


def main(argv=None):
    # Windows Taskbar Icon Fix: Decouple from python.exe so taskbar displays Kinetra's custom icon
    if sys.platform == 'win32':
        try:
            import ctypes
            myappid = 'kinetra.motionstudio.engine.v04'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
    argv = list(sys.argv if argv is None else argv)
    QtCore.QCoreApplication.setOrganizationName(ORG_NAME)
    QtCore.QCoreApplication.setApplicationName(APP_NAME)
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    app = QtWidgets.QApplication(argv)
    app.setStyleSheet(QSS)
    ico_path = ASSETS / 'icon.ico'
    png_path = ASSETS / 'icon.png'
    icon = QtGui.QIcon(str(ico_path if ico_path.exists() else png_path))
    app.setWindowIcon(icon)
    font = QtGui.QFont()
    font.setFamilies(["Segoe UI", "Inter", "Noto Sans", "Noto Sans CJK SC", "Noto Sans CJK JP", "Arial"])
    font.setPointSize(9)
    app.setFont(font)
    win = MainWindow(QtCore.QSettings())
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
