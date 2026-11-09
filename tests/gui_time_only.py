# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Time data only (hammer channel alone) in simulation: hits recorded with an external trigger pulse on
every detected hit, saved without FRFs, then opened in the Data Viewer.
QT_QPA_PLATFORM=offscreen python tests/gui_time_only.py"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox
from pymodalhammer.config import DONE_ACTIONS, TRIGGER_MODES, default_config
from pymodalhammer.gui.main_window import MainWindow
from pymodalhammer.results import find_runs, scan_run, trigger_rows

app = QApplication(sys.argv)
QSettings("pymodalhammer", "pymodalhammer-test").clear()
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
dialogs = []
for n in ("warning", "critical", "information"):
    setattr(QMessageBox, n, staticmethod(lambda *a, **k: dialogs.append(a[1:3]) or print("DIALOG:", a[1:3])
                                         or QMessageBox.Ok))


def pump(ms):
    t = time.perf_counter() + ms / 1000
    while time.perf_counter() < t:
        app.processEvents()
        time.sleep(0.01)


w = MainWindow()
w.show()
w.act_sim.setChecked(True)
tmp = tempfile.mkdtemp(prefix="hammer_time_only_")
cfg = default_config()
for c in cfg.channels[1:]:
    c.enabled = False
cfg.save.folder = tmp
cfg.acq.n_averages = 3
cfg.acq.settle_s = 0.3
cfg.acq.when_done = DONE_ACTIONS[1]                     # stop and save automatically
cfg.trigger.enabled = True
cfg.trigger.mode = TRIGGER_MODES[1]                     # a pulse on every detected hit
w.apply_config(cfg)
assert "time data only" in w.chan.summary.text()
w.start_measurement()
assert not dialogs, dialogs
assert not w.meas.right_tabs.isTabVisible(2) and w.meas.p_r.isHidden()
t0 = time.perf_counter()
while not find_runs(tmp) and time.perf_counter() - t0 < 120:
    pump(100)
pump(1500)
runs = find_runs(tmp)
assert len(runs) == 1, w.meas.log.toPlainText()[-3000:]
run = scan_run(runs[0])
assert run.meta["time_only"] and run.meta["n_averages"] == 3
assert not any(k in run.files for k in ("H1", "H2", "Hv", "Coherence", "AutoSpectra")), run.files
assert "hits" in run.files and "hit_log" in run.files
tr = trigger_rows(run)
assert len(tr) >= 3 and all(t["reason"] == "hit detected" for t in tr), tr
pump(500)
assert w.viewer.tabs.currentIndex() == 1, w.viewer.tabs.currentIndex()      # Time data tab
assert not dialogs, dialogs
print("files:", sorted(os.path.basename(p) for p in run.files.values()))
print("TIME-ONLY OK")
