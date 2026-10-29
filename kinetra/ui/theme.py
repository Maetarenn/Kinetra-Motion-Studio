"""Colours and the application style sheet."""
C_BG, C_PANEL, C_PANEL2, C_TEXT, C_DIM = "#07090E", "#121620", "#0D121B", "#E2E8F0", "#7C8AA0"
C_CYAN, C_AMBER, C_GREEN, C_RED, C_GRID, C_LINE = "#00F0FF", "#FFB300", "#00FF66", "#FF4D5E", "#1A202C", "#1E2A3A"

QSS = f"""
QMainWindow, QDialog, QWidget {{ background:{C_BG}; color:{C_TEXT}; font-size:11px; }}
QLabel, QCheckBox, QRadioButton {{ background:transparent; }}
QToolTip {{ background:{C_PANEL}; color:{C_TEXT}; border:1px solid {C_CYAN}; }}
QMenuBar {{ background:{C_PANEL2}; border-bottom:1px solid {C_LINE}; }}
QMenuBar::item {{ padding:5px 10px; background:transparent; }}
QMenuBar::item:selected {{ background:{C_LINE}; color:{C_CYAN}; }}
QMenu {{ background:{C_PANEL}; border:1px solid {C_LINE}; padding:4px; }}
QMenu::item {{ padding:6px 22px; }}
QMenu::item:selected {{ background:{C_LINE}; color:{C_CYAN}; }}
QMenu::separator {{ height:1px; background:{C_LINE}; margin:4px 8px; }}
QGroupBox {{ background:{C_PANEL}; border:1px solid {C_LINE}; border-radius:5px; margin-top:16px;
             padding:12px 8px 8px 8px; font-weight:bold; color:{C_CYAN}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 5px; letter-spacing:1px; }}
QProgressBar {{ background:{C_BG}; border:1px solid {C_LINE}; border-radius:2px; max-height:10px; min-height:10px; }}
QProgressBar::chunk {{ background:{C_GREEN}; }}
QPushButton {{ background:{C_PANEL}; border:1px solid {C_CYAN}; color:{C_CYAN}; padding:9px 14px;
               font-weight:bold; letter-spacing:1px; border-radius:4px; }}
QPushButton:hover {{ background:{C_CYAN}; color:{C_BG}; }}
QPushButton:pressed {{ background:#00B8C4; color:{C_BG}; }}
QPushButton:disabled {{ border-color:#2A3340; color:#4A5568; background:{C_PANEL}; }}
QPushButton:checked {{ background:{C_GREEN}; border-color:{C_GREEN}; color:{C_BG}; }}
QPushButton#rec:checked {{ background:{C_RED}; border-color:{C_RED}; color:#fff; }}
QPushButton#abort {{ border-color:{C_AMBER}; color:{C_AMBER}; }}
QPushButton#abort:hover {{ background:{C_AMBER}; color:{C_BG}; }}
QPushButton#primary {{ background:{C_CYAN}; color:{C_BG}; }}
QPushButton#primary:hover {{ background:#7FF8FF; }}
QPushButton#primary:disabled {{ background:#1C2A33; color:#4A5568; }}
QPlainTextEdit, QLineEdit, QTextEdit {{ background:#05070B; border:1px solid {C_LINE}; color:{C_GREEN};
                                         selection-background-color:{C_LINE}; }}
QLineEdit {{ color:{C_TEXT}; padding:3px 5px; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background:{C_PANEL2}; border:1px solid {C_LINE}; padding:3px 5px; border-radius:3px; min-height:18px; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color:{C_CYAN}; }}
QComboBox QAbstractItemView {{ background:{C_PANEL}; selection-background-color:{C_LINE}; selection-color:{C_CYAN}; border:1px solid {C_LINE}; }}
QCheckBox::indicator {{ width:14px; height:14px; border:1px solid {C_LINE}; border-radius:3px; background:{C_BG}; }}
QCheckBox::indicator:checked {{ background:{C_CYAN}; border-color:{C_CYAN}; }}
QTabWidget::pane {{ border:1px solid {C_LINE}; top:-1px; }}
QTabBar::tab {{ background:{C_PANEL}; padding:8px 14px; color:{C_DIM}; border:1px solid {C_LINE}; border-bottom:none; letter-spacing:1px; }}
QTabBar::tab:selected {{ color:{C_CYAN}; border-bottom:2px solid {C_CYAN}; background:{C_PANEL2}; }}
QTableWidget {{ background:{C_PANEL2}; gridline-color:{C_LINE}; border:1px solid {C_LINE}; alternate-background-color:{C_PANEL}; }}
QHeaderView::section {{ background:{C_PANEL}; color:{C_CYAN}; border:none; border-right:1px solid {C_LINE}; padding:5px; }}
QTableWidget::item:selected {{ background:{C_LINE}; color:{C_CYAN}; }}
QScrollArea {{ border:none; background:{C_PANEL}; }}
QScrollBar:vertical {{ background:{C_BG}; width:9px; }}
QScrollBar::handle:vertical {{ background:#2A3A52; border-radius:4px; min-height:20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
QSplitter::handle {{ background:{C_LINE}; }}
QStatusBar {{ background:{C_PANEL2}; border-top:1px solid {C_LINE}; }}
"""
