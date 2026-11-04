# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""GUI smoke test in simulation mode (offscreen). Usage:
    QT_QPA_PLATFORM=offscreen python tests/gui_smoke.py <out_dir>
Probe → measurement (auto-save) → manual-review point → removal of a hit → viewer / compare,
saving a screenshot of every page."""

import os
os.environ["PYMODALHAMMER_SETTINGS"] = "pymodalhammer-test"   # never touch the real saved setup
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyqtgraph as pg                                                  # noqa: E402
from PySide6.QtCore import QSettings                  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox                # noqa: E402

from pymodalhammer.config import ACCEPT_MODES, DONE_ACTIONS, TRIGGER_MODES, default_config  # noqa: E402
from pymodalhammer.gui.main_window import MainWindow                    # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hammer_shots"
os.makedirs(out, exist_ok=True)
app = QApplication(sys.argv)
app.setStyle("Fusion")
pg.setConfigOptions(antialias=True, background="w", foreground="k")
QSettings("pymodalhammer", "pymodalhammer-test").clear()
# never block on a dialog in the test: answer Yes / OK
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
for name in ("warning", "critical", "information"):
    setattr(QMessageBox, name, staticmethod(lambda *a, **k: print("DIALOG:", a[1:3]) or QMessageBox.Ok))

w = MainWindow()
w.resize(1500, 950)
w.show()
w.act_sim.setChecked(True)
tmp = tempfile.mkdtemp(prefix="hammer_")
cfg = default_config()
cfg.save.folder = tmp
cfg.acq.n_averages = 4
cfg.acq.settle_s = 0.5
cfg.acq.when_done = DONE_ACTIONS[1]
cfg.trigger.enabled = True
cfg.trigger.mode = TRIGGER_MODES[1]
cfg.save.uff_h2 = True
cfg.save.save_stream_tdms = True
w.apply_config(cfg)


def pump(ms):
    """Process events while letting worker threads run (QTest.qWait may hold the GIL)."""
    import time
    t_end = time.perf_counter() + ms / 1000
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.01)


def shot(widget, name):
    pump(400)
    widget.grab().save(os.path.join(out, name))
    print("shot", name)


def wait_until(cond, timeout_s, what):
    for _ in range(int(timeout_s * 10)):
        if cond():
            return True
        pump(100)
    raise SystemExit(f"TIMEOUT waiting for {what}")


# ---- acquisition setup, then probe mode
w.show_page(w.acq)
shot(w, "01_acquisition_setup.png")
w.show_page(w.chan)
shot(w, "00_channel_setup.png")
w.show_page(w.acq)
w.acq.right.setCurrentIndex(1)
w.start_probe()
wait_until(lambda: len(w.acq.probe._hit_peaks) >= 3, 30, "probe hits")
shot(w, "02_probe_mode.png")
w.stop_engine()
wait_until(lambda: w.worker is None, 10, "probe stop")
w.acq.right.setCurrentIndex(0)
shot(w, "03_windows_with_real_hit.png")

# ---- point 1: automatic, stop & save
w.start_measurement()
wait_until(lambda: w.session is not None and len(w.session.hits) >= 2, 30, "first hits")
w.meas.table.selectRow(0)
w.meas.right_tabs.setCurrentIndex(0)
shot(w, "04_measurement_time.png")
wait_until(lambda: w.worker is None and w.session is None, 60, "point 1 auto-saved")
print("point 1 saved; node now", w.meas.pnode.value())

# ---- point 2: manual review; accept / reject; remove one; save
w.acq.accept.setCurrentText(ACCEPT_MODES[1])
w.acq.done.setCurrentText(DONE_ACTIONS[0])
w.start_measurement()
decisions = 0
while w.session is None or w.session.n_included() < 4:
    wait_until(lambda: w.meas.review.isVisible() or (w.session and w.session.n_included() >= 4), 30, "pending hit")
    if w.meas.review.isVisible():
        if decisions == 1:
            shot(w, "05_manual_review.png")
        w.meas._decide(decisions != 2)          # reject the 3rd one
        decisions += 1
wait_until(lambda: w.worker is None, 20, "stop at completion")
w.meas.table.selectRow(1)
w.meas._delete_selected()                       # delete hit 2
assert all(h.number != 2 for h in w.session.hits)
w.meas.right_tabs.setCurrentIndex(2)
shot(w, "06_measurement_frf.png")
w.meas.right_tabs.setCurrentIndex(1)
shot(w, "07_measurement_spectra.png")
print("point 2 statuses:", [h.status for h in w.session.hits])
assert w.save_current_point()

# ---- viewer
w.show_page(w.viewer)
print("viewer runs:", w.viewer.run_sel.count())
assert w.viewer.run_sel.count() == 2
w.viewer.tabs.setCurrentIndex(0)
shot(w, "08_viewer_frf.png")
w.viewer.tabs.setCurrentIndex(1)
shot(w, "09_viewer_time.png")
w.viewer.tabs.setCurrentIndex(2)
shot(w, "10_viewer_spectra.png")
w.viewer.compare.check([w.viewer.run_sel.itemData(i) for i in range(w.viewer.run_sel.count())])
w.viewer.tabs.setCurrentIndex(3)
w.viewer.compare.item.setCurrentText("Acc_2")
shot(w, "11_viewer_compare.png")
print("compare:", w.viewer.compare.info.text())
cmp = w.viewer.compare
cmp.qty.setCurrentIndex(2)
cmp.item.setCurrentText("Acc_1")
cmp.hitsel.setCurrentIndex(1)
cmp.norm.setChecked(True)
shot(w, "12_viewer_compare_time.png")
print("compare time:", cmp.info.text())
assert "2 run(s)" in cmp.info.text()
cmp.hitsel.setCurrentIndex(3)
cmp.tzoom.setChecked(True)
cmp.item.setCurrentText("Hammer")
print("compare time (mean, hammer):", cmp.info.text())
print("files:", sorted(os.listdir(tmp)))
for d in sorted(os.listdir(tmp)):
    p = os.path.join(tmp, d)
    if os.path.isdir(p):
        print(" ", d, sorted(os.listdir(p)))
w.close()
print("GUI SMOKE OK")
