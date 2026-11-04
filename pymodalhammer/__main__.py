# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Start the GUI:  python -m pymodalhammer   (or the installed command: pymodalhammer)"""

import os
import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def _quiet_qt_noise() -> None:
    """Mute Qt's harmless Wayland text-input focus warnings (GNOME/Wayland)."""
    rules = os.environ.get("QT_LOGGING_RULES", "")
    if "qt.qpa.wayland.textinput" not in rules:
        os.environ["QT_LOGGING_RULES"] = (rules + ";" if rules else "") + "qt.qpa.wayland.textinput=false"


def main() -> int:
    _quiet_qt_noise()
    app = QApplication(sys.argv)
    app.setApplicationName("pymodalhammer")
    app.setStyle("Fusion")
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
