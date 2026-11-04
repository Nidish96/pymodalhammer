# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Probe mode shows the external-trigger pulse. QT_QPA_PLATFORM=offscreen python tests/gui_probe_pulse.py [hw|od]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox
import pyqtgraph as pg
app = QApplication(sys.argv); app.setStyle("Fusion")
pg.setConfigOptions(antialias=True, background="w", foreground="k")
QSettings("pymodalhammer", "pymodalhammer-test").clear()
for n in ("warning", "critical", "information"):
    setattr(QMessageBox, n, staticmethod(lambda *a, **k: print("DIALOG:", a[1:3]) or QMessageBox.Ok))
from pymodalhammer.config import TRIGGER_MODES, TRIGGER_TIMING, default_config
from pymodalhammer.gui.main_window import MainWindow
def pump(ms):
    t = time.perf_counter() + ms / 1000
    while time.perf_counter() < t:
        app.processEvents(); time.sleep(0.01)
w = MainWindow(); w.resize(1500, 950); w.show(); w.act_sim.setChecked(True)
cfg = default_config(); cfg.acq.settle_s = 0.3
cfg.trigger.enabled = True; cfg.trigger.in_probe = True; cfg.trigger.mode = TRIGGER_MODES[1]
cfg.trigger.pulse_samples = 200
timing = sys.argv[1] if len(sys.argv) > 1 else "hw"
if timing == "od": cfg.trigger.timing = TRIGGER_TIMING[1]
w.apply_config(cfg)
w.show_page(w.acq); w.acq.right.setCurrentIndex(1)
w.start_probe()
p = w.acq.probe
t0 = time.perf_counter()
while (p._last_hit is None or not any(r["hit"] == p._last_hit.number for r in p._pulses)) and time.perf_counter() - t0 < 30:
    pump(100)
pump(1200)
print("B =", p.sel_b.currentText(), "| pulses:", [(r["pulse"], r["hit"], r["sample"]) for r in p._pulses])
print("info:", p.pulse_info.text()[:200])
print("markers:", len(p._marks), "trig_wave max:", None if p._last_hit.trig_wave is None else p._last_hit.trig_wave.max())
assert p._pulses and p._marks, "no pulse shown"
print("PROBE PULSE OK")
w.stop_engine(); pump(800)
