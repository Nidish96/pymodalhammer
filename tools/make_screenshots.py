# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerate the manual's screenshots from a simulated session.

    QT_QPA_PLATFORM=offscreen python tools/make_screenshots.py docs/screenshots

Runs probe mode, three roving-hammer points (auto-advance), one manually reviewed point and the
Data Viewer, and saves a PNG of every page, panel and state used in docs/pymodalhammer_manual.org."""

import os
import sys
import tempfile
import time

os.environ["PYMODALHAMMER_SETTINGS"] = "pymodalhammer-screenshots"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyqtgraph as pg                                                      # noqa: E402
from PySide6.QtCore import QSettings                                        # noqa: E402
from PySide6.QtWidgets import QApplication, QGroupBox, QMessageBox          # noqa: E402

from pymodalhammer.config import (ACCEPT_MODES, ASK, DONE_ACTIONS, TRIGGER_MODES,  # noqa: E402
                                  default_config)
from pymodalhammer.gui.main_window import MainWindow                        # noqa: E402

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots")
os.makedirs(OUT, exist_ok=True)
app = QApplication(sys.argv)
app.setStyle("Fusion")
pg.setConfigOptions(antialias=True, background="w", foreground="k")
QSettings("pymodalhammer", "pymodalhammer-screenshots").clear()
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
for _n in ("warning", "critical", "information"):
    setattr(QMessageBox, _n, staticmethod(lambda *a, **k: print("DIALOG:", a[1:3]) or QMessageBox.Ok))


def pump(ms):
    t = time.perf_counter() + ms / 1000
    while time.perf_counter() < t:
        app.processEvents()
        time.sleep(0.01)


def wait(cond, timeout, what):
    t = time.perf_counter() + timeout
    while not cond():
        if time.perf_counter() > t:
            raise SystemExit(f"timeout: {what}")
        pump(100)


def shot(widget, name):
    pump(500)
    widget.grab().save(os.path.join(OUT, name))
    print("  saved", name)


def group(parent, title):
    return next(g for g in parent.findChildren(QGroupBox) if g.title() == title)


w = MainWindow()
w.resize(1600, 1000)
w.show()
w.act_sim.setChecked(True)
tmp = tempfile.mkdtemp(prefix="pymodalhammer_shots_")
cfg = default_config()
cfg.save.folder = tmp
cfg.acq.settle_s = 0.5
cfg.acq.n_averages = 3
cfg.trigger.enabled = True
cfg.trigger.in_probe = True
cfg.trigger.mode = TRIGGER_MODES[1]
cfg.trigger.pulse_samples = 100
cfg.save.uff_h2 = True
w.apply_config(cfg)

# ------------------------------------------------------------ setup tabs
w.show_page(w.chan)
shot(w, "01_channel_setup.png")
w.show_page(w.acq)
w.acq.right.setCurrentIndex(0)
shot(w, "02_acquisition_setup.png")
for title, name in (("Record", "03a_group_record.png"), ("Trigger (hit detection)", "03b_group_trigger.png"),
                    ("Windows", "03c_group_windows.png"), ("Averaging && acceptance", "03d_group_averaging.png"),
                    ("Hit checks — what to do with a bad hit", "03e_group_checks.png"),
                    ("External trigger (e.g. high-speed camera)", "03f_group_external_trigger.png")):
    shot(group(w.acq, title), name)

# ------------------------------------------------------------ probe mode
w.acq.right.setCurrentIndex(1)
w.start_probe()
p = w.acq.probe
wait(lambda: len(p._hit_peaks) >= 3 and p._last_hit is not None and
     any(r["hit"] == p._last_hit.number for r in p._pulses), 60, "probe hits")
pump(800)
shot(w, "04_probe_pulse.png")
p.sel_b.setCurrentIndex(2)                     # Acc_1
shot(w, "05_probe_mode.png")
w.stop_engine()
wait(lambda: w.worker is None, 10, "probe stop")
w.acq.right.setCurrentIndex(0)
w.acq.zoom.setChecked(True)
shot(w.acq.right, "06_windows_zoom_real_hit.png")
w.acq.zoom.setChecked(False)
shot(w.acq.right, "06b_windows_full_record.png")

# ------------------------------------------------------------ three roving-hammer points, auto-advance
w.acq.done.setCurrentText(DONE_ACTIONS[2])
w.show_page(w.meas)
shot(group(w.meas, "Save results"), "07a_group_save.png")
shot(group(w.meas, "Measurement point"), "07b_group_point.png")
w.start_measurement()
wait(lambda: w.session is not None and len(w.session.hits) >= 2, 60, "point 1 hits")
w.meas.right_tabs.setCurrentIndex(0)
shot(w, "08_measurement_armed.png")
from pymodalhammer.results import find_runs                                # noqa: E402
wait(lambda: len(find_runs(tmp)) >= 3, 180, "three points")
w.stop_engine()
wait(lambda: w.worker is None, 10, "stop")
print("  points saved:", [os.path.basename(r) for r in find_runs(tmp)])

# ------------------------------------------------------------ point 4: a double hit (Ask me) and an overload
# The simulator is steered so that the pictures show what the manual describes: clean hits, one double
# hit held for a decision (accepted → flagged), and one overload (rejected automatically).
w.acq.accept.setCurrentText(ACCEPT_MODES[0])
w.acq.done.setCurrentText(DONE_ACTIONS[0])
w.acq.navg.setValue(4)
w.acq.dbl_act.setCurrentText(ASK)
w.start_measurement()
wait(lambda: w.worker is not None and w.worker.engine is not None
     and hasattr(w.worker.engine.backend, "double_prob"), 20, "engine")
sim = w.worker.engine.backend
sim.double_prob, sim.overload_prob = 0.0, 0.0
wait(lambda: w.session is not None and len(w.session.hits) >= 1, 60, "first clean hit")
sim.double_prob = 1.0                                   # next hit bounces
wait(lambda: not w.meas.review.isHidden(), 60, "double hit pending")
sim.double_prob = 0.0
w.meas.right_tabs.setCurrentIndex(0)
shot(w, "09_manual_review.png")
w.meas._decide(True)                                    # accept it → flagged
sim.overload_prob = 1.0                                 # next hit overloads → rejected automatically
n0 = len(w.session.hits)
wait(lambda: len(w.session.hits) > n0, 60, "overload hit")
sim.overload_prob = 0.0
wait(lambda: w.worker is None, 30, "stop at completion")
w.meas.table.selectRow(0)
w.meas.right_tabs.setCurrentIndex(0)
shot(w, "10_measurement_time.png")
w.meas.right_tabs.setCurrentIndex(1)
shot(w, "11_measurement_spectra.png")
w.meas.table.clearSelection()
w.meas.right_tabs.setCurrentIndex(2)
shot(w, "12_measurement_frf.png")
w.meas.table.selectRow(1)
shot(w.meas.table.parentWidget(), "13_hit_list.png")
assert w.save_current_point()

# ------------------------------------------------------------ data viewer
w.show_page(w.viewer)
pump(500)
v = w.viewer
v.tabs.setCurrentIndex(0)
shot(w, "14_viewer_frf.png")
v.tabs.setCurrentIndex(1)
v.hits.setCurrentRow(0)
shot(w, "15_viewer_time.png")
v.tabs.setCurrentIndex(2)
shot(w, "16_viewer_spectra.png")
v.tabs.setCurrentIndex(0)
v.rp_end.setValue(1.0)
v._apply_windows()
shot(w, "17_viewer_reprocess.png")
v._reset_windows()
c = v.compare
c.check([v.run_sel.itemData(i) for i in range(v.run_sel.count())])
v.tabs.setCurrentIndex(3)
c.qty.setCurrentIndex(0)
c.item.setCurrentText("Acc_2")
shot(w, "18_compare_frf.png")
c.qty.setCurrentIndex(2)
c.item.setCurrentText("Acc_2")
c.hitsel.setCurrentIndex(0)
c.norm.setChecked(True)
shot(w, "19_compare_time.png")
c.tzoom.setChecked(True)
c.item.setCurrentText("Hammer")
c.hitsel.setCurrentIndex(1)
shot(w, "20_compare_time_hammer.png")
c.tzoom.setChecked(False)
c.qty.setCurrentIndex(1)
c.item.setCurrentText("Hammer")
shot(w, "21_compare_spectrum.png")
w.close()
print("files of one point:", sorted(os.listdir(find_runs(tmp)[-1])))
print("DONE", OUT)
