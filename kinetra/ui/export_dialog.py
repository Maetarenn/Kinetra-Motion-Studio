"""Dialog presented after recording a take with direct links and instructions for Blender and Maya."""
import subprocess
import sys
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import t
from .theme import C_BG, C_PANEL, C_CYAN, C_GREEN, C_AMBER, C_TEXT, C_DIM


class TakeExportDialog(QtWidgets.QDialog):
    def __init__(self, take_dir, parent=None):
        super().__init__(parent)
        self.take_dir = Path(take_dir)
        self.setWindowTitle(t("export.dlg_title"))
        self.setMinimumSize(560, 480)
        self.setStyleSheet(f"background-color: {C_BG}; color: {C_TEXT};")
        self._build_ui()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(14)
        lay.setContentsMargins(18, 18, 18, 18)

        # Header with green checkmark / success banner
        hdr = QtWidgets.QHBoxLayout()
        icon_lbl = QtWidgets.QLabel("🎬")
        icon_lbl.setStyleSheet("font-size: 32px;")
        hdr.addWidget(icon_lbl)

        txt_col = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(t("export.success_title"))
        title.setStyleSheet(f"color: {C_GREEN}; font-size: 18px; font-weight: bold;")
        sub = QtWidgets.QLabel(f"{t('export.take_folder')}: {self.take_dir.name}")
        sub.setStyleSheet(f"color: {C_DIM}; font-size: 12px;")
        txt_col.addWidget(title)
        txt_col.addWidget(sub)
        hdr.addLayout(txt_col, 1)
        lay.addLayout(hdr)

        # File list box
        g = QtWidgets.QGroupBox(t("export.files_created"))
        g.setStyleSheet(f"QGroupBox {{ background-color: {C_PANEL}; border-radius: 6px; padding: 10px; font-weight: bold; color: {C_CYAN}; }}")
        glay = QtWidgets.QVBoxLayout(g)

        bvh_file = self.take_dir / "motion.bvh"
        usd_file = self.take_dir / "motion.usda"
        csv_file = self.take_dir / "face_curves.csv"

        for label, p, color in (
            ("Animasyon İskeleti (BVH):", bvh_file, C_GREEN),
            ("Evrensel Sahne (USD):", usd_file, C_CYAN),
            ("Yüz Blendshape Eğrileri (CSV):", csv_file, C_AMBER),
        ):
            if p.exists():
                row = QtWidgets.QHBoxLayout()
                lbl = QtWidgets.QLabel(label)
                lbl.setStyleSheet("font-weight: bold;")
                val = QtWidgets.QLabel(p.name)
                val.setStyleSheet(f"color: {color}; font-weight: bold;")
                size_str = f"({p.stat().st_size / 1024:.1f} KB)"
                sz = QtWidgets.QLabel(size_str)
                sz.setStyleSheet(f"color: {C_DIM}; font-size: 11px;")
                row.addWidget(lbl)
                row.addWidget(val)
                row.addWidget(sz)
                row.addStretch(1)
                glay.addLayout(row)
        lay.addWidget(g)

        # Integration Guide Tabs (Blender & Maya)
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(f"QTabWidget::pane {{ border: 1px solid {C_DIM}; background: {C_PANEL}; border-radius: 4px; }} "
                           f"QTabBar::tab {{ background: {C_BG}; color: {C_TEXT}; padding: 6px 14px; font-weight: bold; }} "
                           f"QTabBar::tab:selected {{ background: {C_PANEL}; color: {C_CYAN}; }}")

        # Blender tab
        blender_txt = QtWidgets.QTextBrowser()
        blender_txt.setStyleSheet("border: none; padding: 10px; font-size: 12px; line-height: 1.5;")
        blender_txt.setHtml(f"""
            <h3 style="color:{C_CYAN}; margin-top:0;">Blender İçe Aktarma Rehberi</h3>
            <ol style="margin-left: -15px;">
                <li>Blender'ı açın ve boş bir sahne oluşturun.</li>
                <li><b>File &gt; Import &gt; Motion Capture (.bvh)</b> seçeneğine tıklayın.</li>
                <li>Bu klasördeki <b>motion.bvh</b> dosyasını seçip içe aktarın.</li>
                <li>İskelet tüm hareket yakalama animasyonu ile sahnenize eklenir!</li>
                <li><i>Karakterinize aktarmak için Blender'ın yerleşik 'Rokoko Studio Live' veya 'Auto-Rig Pro / Rigify' eklentilerini kullanabilirsiniz.</i></li>
            </ol>
        """)
        tabs.addTab(blender_txt, "Blender")

        # Maya tab
        maya_txt = QtWidgets.QTextBrowser()
        maya_txt.setStyleSheet("border: none; padding: 10px; font-size: 12px; line-height: 1.5;")
        maya_txt.setHtml(f"""
            <h3 style="color:{C_CYAN}; margin-top:0;">Autodesk Maya İçe Aktarma Rehberi</h3>
            <ol style="margin-left: -15px;">
                <li>Maya'yı açın.</li>
                <li><b>File &gt; Import</b> menüsüne gidin ve <b>motion.bvh</b> dosyasını seçin.</li>
                <li>Zaman çizelgesini (Time Slider) oynatarak animasyonu izleyin.</li>
                <li>Kendi 3D karakterinizin <b>HumanIK</b> penceresinde <b>Source = motion</b> olarak seçin.</li>
                <li>Kinetra'nın yakaladığı hareket karakterinize birebir aktarılacaktır!</li>
            </ol>
        """)
        tabs.addTab(maya_txt, "Autodesk Maya")

        lay.addWidget(tabs, 1)

        # Bottom Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_open = QtWidgets.QPushButton(t("export.btn_open_folder"))
        btn_open.setStyleSheet(f"background-color: {C_CYAN}; color: #000; font-weight: bold; padding: 8px 16px; border-radius: 4px;")
        btn_open.clicked.connect(self._open_in_explorer)
        btn_box.addWidget(btn_open)

        btn_box.addStretch(1)

        btn_ok = QtWidgets.QPushButton(t("dlg.ok"))
        btn_ok.setStyleSheet("padding: 8px 24px; font-weight: bold; border-radius: 4px;")
        btn_ok.clicked.connect(self.accept)
        btn_box.addWidget(btn_ok)

        lay.addLayout(btn_box)

    def _open_in_explorer(self):
        bvh = self.take_dir / "motion.bvh"
        target = bvh if bvh.exists() else self.take_dir
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{target}"])
        else:
            subprocess.Popen(["xdg-open", str(self.take_dir)])
