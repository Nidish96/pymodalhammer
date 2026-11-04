# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Roving-hammer auto-advance in simulation: 3 points saved without stopping, with an
'averages complete' trigger pulse and Ask-me double hits.  QT_QPA_PLATFORM=offscreen python tests/gui_autoadvance.py"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox
from pymodalhammer.config import ASK, DONE_ACTIONS, TRIGGER_MODES, default_config
from pymodalhammer.gui.main_window import MainWindow
from pymodalhammer.results import find_runs, scan_run, trigger_rows

app = QApplication(sys.argv)
QSettings("pymodalhammer", "pymodalhammer-test").clear()
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
for n in ("warning", "critical", "information"):
    setattr(QMessageBox, n, staticmethod(lambda *a, **k: print("DIALOG:", a[1:3]) or QMessageBox.Ok))


def pump(ms):
    t = time.perf_counter() + ms / 1000
    while time.perf_counter() < t:
        app.processEvents()
        time.sleep(0.01)


w = MainWindow()
w.show()
w.act_sim.setChecked(True)
tmp = tempfile.mkdtemp(prefix="hammer_adv_")
cfg = default_config()
cfg.save.folder = tmp
cfg.acq.n_averages = 2
cfg.acq.settle_s = 0.3
cfg.acq.double_hit_action = ASK
cfg.acq.when_done = DONE_ACTIONS[2]
cfg.trigger.enabled = True
cfg.trigger.mode = TRIGGER_MODES[4]
w.apply_config(cfg)
w.start_measurement()
t0 = time.perf_counter()
while len(find_runs(tmp)) < 3 and time.perf_counter() - t0 < 120:
    pump(100)
    if not w.meas.review.isHidden():
        print("ask-me hit → reject")
        w.meas._decide(False)
w.stop_engine()
pump(1500)
runs = find_runs(tmp)
print("runs:", [os.path.basename(r) for r in runs])
if len(runs) < 3:
    print(w.meas.log.toPlainText()[-3000:])
    print("phase:", w.meas.phase.text(), "| session:", w.session and [(h.number, h.status) for h in w.session.hits])
assert len(runs) >= 3
for r in runs[:3]:
    run = scan_run(r)
    tr = trigger_rows(run)
    print(run.point_label, run.meta["n_averages"], "avg; pulses:", [(t["reason"], t["sample_index"]) for t in tr])
    assert any(t["reason"] == "point complete" for t in tr)
print("node now", w.meas.pnode.value())
print("AUTO-ADVANCE OK")
