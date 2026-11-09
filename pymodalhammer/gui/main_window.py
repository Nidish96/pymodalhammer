# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Main window: tabs, menus, hardware/simulation mode, the acquisition thread (probe or measurement)
and the life cycle of a measurement point (hits → average → save → next point)."""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import time
import traceback

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QLabel, QMainWindow, QMessageBox, QScrollArea,
                               QSizePolicy, QTabWidget)

from ..backends import discover_devices, make_backend
from ..config import DONE_ACTIONS, AppConfig, default_config
from ..engine import M_ACCEPT, M_DONE, Callbacks, HammerEngine
from ..export import point_folder, save_point
from ..processing import PointSession
from .acquisition_setup import AcquisitionSetup
from .channel_setup import ChannelTable
from .measurement import MeasurementTab
from .viewer import ViewerTab


class HammerWorker(QObject):
    """Runs HammerEngine in a QThread and turns its callbacks into Qt signals."""
    log = Signal(str)
    scope = Signal(object, int, float, object, int, object)
    hit = Signal(object)
    pulse = Signal(object)
    started = Signal(float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: AppConfig, simulate: bool, probe: bool, segment: int = 1, first_hit: int = 1,
                 stream_path: str = "", sim_auto: bool = True):
        super().__init__()
        self.cfg, self.simulate, self.probe = cfg, simulate, probe
        self.segment, self.first_hit, self.stream_path = segment, first_hit, stream_path
        self.sim_auto = sim_auto
        self.engine = None

    @Slot()
    def run(self):
        summary = None
        try:
            kw = {"auto_hits": self.sim_auto} if self.simulate else {}
            backend = make_backend(self.cfg, self.simulate, probe=self.probe, **kw)
            cb = Callbacks(log=self.log.emit, scope=self.scope.emit, hit=self.hit.emit, pulse=self.pulse.emit,
                           started=self.started.emit)
            self.engine = HammerEngine(self.cfg, backend, cb, probe=self.probe, segment=self.segment,
                                       first_hit_number=self.first_hit, stream_path=self.stream_path)
            summary = self.engine.run()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}")
        self.finished.emit(summary)

    # thread-safe pass-throughs (the engine only sets flags / queues)
    def stop(self):
        if self.engine is not None:
            self.engine.request_stop()

    def rearm(self):
        if self.engine is not None:
            self.engine.rearm()

    def disarm(self):
        if self.engine is not None:
            self.engine.disarm()

    def request_pulse(self, reason: str, hit: int = 0):
        if self.engine is not None:
            self.engine.request_pulse(reason, hit)

    def new_point(self, node: int):
        if self.engine is not None:
            self.engine.new_point(node)

    def simulate_hit(self):
        if self.engine is not None and hasattr(self.engine.backend, "request_hit"):
            self.engine.backend.request_hit()


class PageScroll(QScrollArea):
    """A tab page that always fills the window and scrolls only when the window is smaller than the
    page's minimum size. (QScrollArea's own resizing follows the preferred height of pages with
    word-wrapped labels, which made pages taller than the window.)"""

    def __init__(self, page):
        super().__init__()
        self.setWidget(page)
        self.setWidgetResizable(False)
        self.setFrameShape(QFrame.NoFrame)
        page.installEventFilter(self)

    def _fit(self):
        page = self.widget()
        vp, m = self.maximumViewportSize(), page.minimumSizeHint()
        page.resize(max(vp.width(), m.width()), max(vp.height(), m.height()))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._fit()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.LayoutRequest:
            self._fit()
        return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from .. import __version__
        self.setWindowTitle(f"pymodalhammer {__version__} — impact-hammer modal testing with NI-DAQmx")
        self.resize(1450, 900)
        self.settings = QSettings("pymodalhammer", os.environ.get("PYMODALHAMMER_SETTINGS", "pymodalhammer"))
        self.current_file = ""

        self.chan = ChannelTable()
        self.acq = AcquisitionSetup()
        self.meas = MeasurementTab()
        self.viewer = ViewerTab()
        self.tabs = QTabWidget()
        self._pages = {}
        for wdg, title in ((self.chan, "1 · Channel Setup"), (self.acq, "2 · Acquisition Setup"),
                           (self.meas, "3 · Measurement"), (self.viewer, "4 · Data Viewer")):
            # every page scrolls if the screen is smaller than its content, so the window can always be
            # maximised (no scroll bars appear when there is room)
            sc = PageScroll(wdg)
            self._pages[wdg] = sc
            self.tabs.addTab(sc, title)
        self.tabs.setStyleSheet("QTabBar::tab { padding: 8px 18px; font-size: 10.5pt }")
        self.setCentralWidget(self.tabs)
        self._viewer_pending = None
        self.tabs.currentChanged.connect(self._tab_changed)
        self.mode_label = QLabel()
        self.mode_label.setMinimumWidth(10)
        self.mode_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.statusBar().addPermanentWidget(self.mode_label, 1)
        self._cap_combo_widths()

        m = self.menuBar().addMenu("&File")
        self._act(m, "&New setup", self.new_setup, QKeySequence.New)
        self._act(m, "&Open setup…", self.open_setup, QKeySequence.Open)
        self._act(m, "&Save setup", self.save_setup, QKeySequence.Save)
        self._act(m, "Save setup &as…", self.save_setup_as, QKeySequence.SaveAs)
        m.addSeparator()
        self._act(m, "&Quit", self.close, QKeySequence.Quit)
        h = self.menuBar().addMenu("&Hardware")
        self._act(h, "&Refresh NI devices", self.refresh_devices, "F5")
        self.act_sim = QAction("&Simulation mode (no hardware)", self, checkable=True)
        self.act_sim.toggled.connect(self._update_mode_label)
        h.addAction(self.act_sim)
        self.act_sim_auto = QAction("Simulator: random hits every 1.6–3.2 s", self, checkable=True)
        self.act_sim_auto.setChecked(True)
        self.act_sim_auto.setToolTip("Off: the simulated hammer only strikes when you press 'Simulate a hit'")
        h.addAction(self.act_sim_auto)

        # ---------------- wiring
        self.chan.changed.connect(self._update_channel_names)
        p = self.acq.probe
        p.start_requested.connect(self.start_probe)
        p.stop_requested.connect(self.stop_engine)
        p.simulate_hit.connect(self._simulate_hit)
        p.suggest_ranges.connect(self._apply_ranges)
        self.acq.test_pulse_requested.connect(self.test_pulse)
        self.meas.start_requested.connect(self.start_measurement)
        self.meas.stop_requested.connect(self.stop_engine)
        self.meas.save_requested.connect(self.save_current_point)
        self.meas.discard_requested.connect(self.discard_point)
        self.meas.decision.connect(self._on_decision)
        self.meas.changed_session.connect(self._check_complete)
        self.meas.btn_sim.clicked.connect(self._simulate_hit)

        self.thread = None
        self.worker = None
        self.mode = None                      # None | 'probe' | 'measure'
        self.session = None
        self.session_cfg = None
        self.session_key = ""
        self.pulses = []
        self.segment = 0
        self._completing = False
        self._save_after_stop = False
        self._t_start = 0.0
        self._fs = 0.0

        self.refresh_devices(first=True)
        saved = self.settings.value("last_setup", "")
        try:
            cfg = AppConfig.from_json(saved) if saved else default_config()
        except Exception:
            cfg = default_config()
        if not cfg.save.folder:
            cfg.save.folder = os.path.join(os.path.expanduser("~"), "pymodalhammer_data")
        self.apply_config(cfg)
        self.viewer.folder.setText(cfg.save.folder)

    def _tab_changed(self, *_):
        if self._viewer_pending and self.tabs.currentWidget() is self._pages.get(self.viewer):
            folder, parent = self._viewer_pending
            self._viewer_pending = None
            self.viewer.open_folder(folder, parent=parent)

    def show_page(self, page):
        self.tabs.setCurrentWidget(self._pages.get(page, page))

    def _cap_combo_widths(self):
        """Drop-downs must not grow with their longest entry (run names, channel labels, long option
        texts), or the window's minimum width exceeds the screen and it cannot be maximised. The open
        list is wide enough to read every entry, and the tooltip shows the current one."""
        for cb in self.findChildren(QComboBox):
            if cb.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon:
                continue
            if cb.isEditable() or cb.sizeHint().width() > 150 or cb.count() == 0:   # long or filled later
                cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                cb.setMinimumContentsLength(10)
                cb.view().setMinimumWidth(320)
                cb.currentTextChanged.connect(cb.setToolTip)

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # ------------------------------------------------------------------ devices
    def refresh_devices(self, first=False):
        info = discover_devices()
        if first:
            self.act_sim.setChecked(not (info["ok"] and info["devices"]))
        if not info["ai"]:
            info["ai"] = [f"cDAQ1Mod{m}/ai{i}" for m in (1, 3) for i in range(4)]
            info["ao"] = [f"cDAQ1Mod2/ao{i}" for i in range(4)]
            info["do"] = [f"cDAQ1Mod4/port0/line{i}" for i in range(8)]
            info["pfi"] = [f"/cDAQ1Mod4/PFI{i}" for i in range(8)]
        self.chan.set_ai_names(info["ai"])
        self.acq.set_output_names(info["ao"], info["do"], info["pfi"])
        self.hw_message = info["message"]
        self._update_mode_label()
        self.statusBar().showMessage(info["message"], 8000)

    def _update_mode_label(self, *_):
        sim = self.act_sim.isChecked()
        self.mode_label.setText("<b style='color:#d35400'>SIMULATION MODE</b> — no hardware is used" if sim else
                                f"<b style='color:#1e8449'>HARDWARE</b> — {self.hw_message}")
        self.meas.btn_sim.setVisible(sim)

    # ------------------------------------------------------------------ config
    def _update_channel_names(self):
        en = [c for c in self.chan.get_channels() if c.enabled]
        ref = next((c.name for c in en if c.is_reference), "")
        self.acq.set_channel_names([(c.name, c.units, c.max_range) for c in en], ref)

    def gather_config(self) -> AppConfig:
        return AppConfig(channels=self.chan.get_channels(), acq=self.acq.get_config(), trigger=self.acq.get_trigger(),
                         save=self.meas.get_save(), null_bridges=self.chan.null_bridges.isChecked())

    def apply_config(self, cfg: AppConfig):
        self.chan.set_channels(cfg.channels)
        self.chan.null_bridges.setChecked(cfg.null_bridges)
        self._update_channel_names()
        self.acq.set_config(cfg.acq)
        self.acq.set_trigger(cfg.trigger)
        self.meas.set_save(cfg.save)

    def new_setup(self):
        cfg = default_config()
        cfg.save.folder = self.meas.folder.text()
        self.apply_config(cfg)
        self.current_file = ""

    def open_setup(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open setup", "", "Setup (*.json)")
        if p:
            try:
                with open(p, encoding="utf-8") as f:
                    self.apply_config(AppConfig.from_json(f.read()))
                self.current_file = p
                self.statusBar().showMessage(f"Loaded {p}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Open setup", f"Could not read setup:\n{e}")

    def save_setup(self):
        if not self.current_file:
            return self.save_setup_as()
        with open(self.current_file, "w", encoding="utf-8") as f:
            f.write(self.gather_config().to_json())
        self.statusBar().showMessage(f"Saved {self.current_file}", 5000)

    def save_setup_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save setup", "hammer_setup.json", "Setup (*.json)")
        if p:
            self.current_file = p
            self.save_setup()

    @staticmethod
    def _key(cfg: AppConfig) -> str:
        """What must not change within one point (channels, record, windows, point)."""
        d = json.loads(cfg.to_json())
        s = d["save"]
        return json.dumps({"ch": d["channels"], "acq": {k: v for k, v in d["acq"].items()
                                                       if k not in ("n_averages", "accept_mode", "when_done",
                                                                    "probe_headroom", "probe_trig_pct")},
                           "pt": [s["point_mode"], s["point_node"], s["point_dir"]]}, sort_keys=True)

    # ------------------------------------------------------------------ worker
    def _launch(self, cfg: AppConfig, probe: bool, stream_path: str = "", first_hit: int = 1):
        sim = self.act_sim.isChecked()
        self.thread = QThread(self)
        self.worker = HammerWorker(cfg, sim, probe, self.segment, first_hit, stream_path,
                                   sim_auto=self.act_sim_auto.isChecked())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        q = Qt.ConnectionType.QueuedConnection
        self.worker.log.connect(self.meas.append_log, q)
        self.worker.failed.connect(self._on_failed, q)
        self.worker.started.connect(self._on_started, q)
        if probe:
            self.worker.scope.connect(self.acq.probe.on_scope, q)
            self.worker.pulse.connect(self.acq.probe.on_pulse, q)
            self.worker.hit.connect(self._on_probe_hit, q)
            self.worker.log.connect(self._status_msg, q)
        else:
            self.worker.scope.connect(self.meas.on_scope, q)
            self.worker.scope.connect(self._status_from_scope, q)
            self.worker.hit.connect(self._on_hit, q)
            self.worker.pulse.connect(self._on_pulse, q)
        self.worker.finished.connect(self._on_finished, q)
        self.worker.finished.connect(self.thread.quit, q)
        self.thread.finished.connect(self.worker.deleteLater)
        self._t_start = time.perf_counter()
        self.thread.start()

    @Slot(str)
    def _status_msg(self, s: str):
        self.statusBar().showMessage(s, 6000)

    def _busy(self) -> bool:
        if self.worker is not None:
            QMessageBox.information(self, "Busy", "The acquisition is running. Stop it first.")
            return True
        return False

    # ------------------------------------------------------------------ probe
    def start_probe(self):
        if self._busy():
            return
        cfg = self.gather_config()
        errs = cfg.validate(probe=True)
        if errs:
            QMessageBox.warning(self, "Check the setup", "Please fix the following:\n\n• " + "\n• ".join(errs))
            return
        self.mode = "probe"
        self.acq.probe.prepare(cfg.labelled_channels(), cfg.reference_index(), cfg.trigger_index(),
                               self.act_sim.isChecked(), cfg.trigger)
        self.acq.probe.set_running(True)
        self.meas.append_log(f"---- {time.strftime('%H:%M:%S')} probe mode ----")
        self._set_editing_enabled(False)
        self._launch(cfg, probe=True)

    @Slot(object)
    def _on_probe_hit(self, hit):
        self.acq.probe.on_hit(hit)
        ri = self.gather_config().reference_index()
        resp = next((i for i in range(hit.data.shape[0]) if i != ri), -1)
        self.acq.set_example_hit(hit.data, hit.pre, hit.fs, ri, resp)

    def _apply_ranges(self, ranges: dict):
        if QMessageBox.question(self, "Apply ranges", "Set these input ranges on the Channel Setup tab?\n\n"
                                + "\n".join(f"{k}: ±{v:g}" for k, v in ranges.items())) == QMessageBox.Yes:
            self.chan.set_ranges(ranges)

    # ------------------------------------------------------------------ measurement
    def start_measurement(self):
        if self._busy():
            return
        cfg = self.gather_config()
        errs = cfg.validate()
        if errs:
            QMessageBox.warning(self, "Check the setup", "Please fix the following:\n\n• " + "\n• ".join(errs))
            return
        key = self._key(cfg)
        if self.session is not None and self.session.hits and key != self.session_key:
            r = QMessageBox.question(self, "Setup changed",
                                     f"Channels, record, trigger, windows or the point changed since this point's first "
                                     f"hit. Its {len(self.session.hits)} hit(s) cannot be averaged with new ones.\n\n"
                                     "Discard them and start the point again?")
            if r != QMessageBox.Yes:
                return
            self.session = None
        if self.session is None:
            self._new_session(cfg)
        else:
            self.session.acq = cfg.acq                     # averages / acceptance may change between segments
            self.meas.progress.setMaximum(cfg.acq.n_averages)
        self.session_cfg.acq = copy.deepcopy(cfg.acq)
        self.session_cfg.trigger = copy.deepcopy(cfg.trigger)
        self.segment += 1
        self.session.segment = self.segment
        stream = ""
        if cfg.save.save_stream_tdms:
            os.makedirs(cfg.save.folder, exist_ok=True)
            stream = os.path.join(cfg.save.folder, f"{cfg.save.basename}_stream_{_dt.datetime.now():%Y%m%d_%H%M%S}.tdms")
            self.meas.append_log(f"Continuous record → {stream}")
        self.mode = "measure"
        self._completing = False
        self._save_after_stop = False
        self.meas.append_log(f"---- {time.strftime('%H:%M:%S')} start, segment {self.segment} "
                             f"({'simulation' if self.act_sim.isChecked() else 'hardware'}), point "
                             f"{self._point_text(cfg)} ----")
        if cfg.time_only:
            self.meas.append_log("Only the hammer channel is enabled: time data only, no FRFs are computed or saved.")
        self.settings.setValue("last_setup", cfg.to_json())
        self._set_editing_enabled(False)
        self.meas.set_running(True)
        self.meas.set_phase("Starting…")
        self.show_page(self.meas)
        self._launch(cfg, probe=False, stream_path=stream, first_hit=len(self.session.hits) + 1)

    def _new_session(self, cfg: AppConfig):
        self.session = PointSession(cfg.labelled_channels(), cfg.reference_index(), cfg.acq, cfg.acq.fs)
        self.session_cfg = copy.deepcopy(cfg)
        self.session_key = self._key(cfg)
        self.pulses = []
        self.segment = 0
        self.meas.set_session(self.session, cfg.acq.trig_level)

    @staticmethod
    def _point_text(cfg: AppConfig) -> str:
        s = cfg.save
        return f"hammer at node {s.point_node}{s.point_dir}" if s.roving else "fixed hammer"

    @Slot(float)
    def _on_started(self, fs: float):
        self._fs = fs
        self._t_start = time.perf_counter()
        if self.mode == "measure" and self.session is not None:
            self.session.set_fs(fs)

    @Slot(object, int, float, object, int, object)
    def _status_from_scope(self, data, start, fs, peaks, backlog, trig=None):
        if self.session is None or self.mode != "measure":
            return
        t = (start + data.shape[1]) / fs
        a = self.session_cfg.acq
        n, N = self.session.n_included(), a.n_averages
        if self._completing:
            return
        if self.session.pending():
            self.meas.set_phase("Waiting for your decision (Accept / Reject)")
        elif t < a.settle_s:
            self.meas.set_phase(f"Settling sensors… {a.settle_s - t:.1f} s")
        else:
            what = "hits" if self.session.time_only else "averages"
            self.meas.set_phase(f"ARMED — hit the structure ({n} of {N} {what})  ·  "
                                f"{self._point_text(self.session_cfg)}")

    @Slot(object)
    def _on_hit(self, hit):
        if self.session is None:
            return
        c = self.session.channels[self.session.ref_i]
        self.meas.append_log(f"Hit {hit.number}: {hit.status}, hammer {hit.peaks[self.session.ref_i]:.4g} {c.units}"
                             + (f" — {hit.reason}" if hit.reason else ""))
        self.meas.on_hit(hit)
        ri = self.session.ref_i
        resp = next((i for i in range(hit.data.shape[0]) if i != ri), -1)
        self.acq.set_example_hit(hit.data, hit.pre, hit.fs, ri, resp)

    @Slot(object)
    def _on_pulse(self, rec: dict):
        self.pulses.append(rec)
        self.meas.append_log(f"  trigger pulse {len(self.pulses)} ({rec['reason']}) at sample {rec['sample']}"
                             + ("" if rec["exact"] else " (approx.)")
                             + f", {rec['latency_ms']:.0f} ms after the request")

    def _on_decision(self, number: int, accepted: bool):
        self.meas.append_log(f"Hit {number}: {'accepted' if accepted else 'rejected'} by you")
        if self.worker is not None:
            if accepted and self.session_cfg.trigger.enabled and self.session_cfg.trigger.mode_index() == M_ACCEPT:
                self.worker.request_pulse("accepted hit", number)
            if not self.session.pending() and not self.session.complete():
                self.worker.rearm()

    def _check_complete(self):
        s = self.session
        if s is None or self.mode != "measure" or self.worker is None or self._completing:
            return
        if not s.complete() or s.pending():
            return
        self._completing = True
        cfg = self.session_cfg
        self.worker.disarm()
        delay = 0
        t = cfg.trigger
        if t.enabled and t.mode_index() == M_DONE:
            self.worker.request_pulse("point complete", 0)
            delay = int(cfg.acq.out_queue_ms + 3 * cfg.acq.chunk_ms) + 50
        done = cfg.acq.when_done
        self.meas.append_log(f"Point complete: {s.n_included()} averages.")
        if done == DONE_ACTIONS[2]:                       # save, then the next point without stopping
            QTimer.singleShot(delay, self._save_and_next)
        else:
            self._save_after_stop = done == DONE_ACTIONS[1]
            self.meas.set_phase("Complete — stopping…")
            QTimer.singleShot(delay, self.stop_engine)

    def _save_and_next(self):
        if self.save_current_point(auto=True):
            cfg = self.gather_config()                     # point node already advanced
            self.segment_keep = self.segment
            self._new_session(cfg)
            self.segment = self.segment_keep
            self.session.segment = self.segment
            self._completing = False
            if self.worker is not None:
                self.worker.new_point(cfg.save.point_node)
            self.meas.append_log(f"Next point: {self._point_text(cfg)} — armed.")
        else:
            self.stop_engine()

    def stop_engine(self):
        if self.worker is not None:
            self.meas.btn_stop.setEnabled(False)
            self.acq.probe.btn_stop.setEnabled(False)
            self.worker.stop()

    def save_current_point(self, auto: bool = False) -> bool:
        s = self.session
        if s is None or s.n_included() == 0:
            if not auto:
                QMessageBox.information(self, "Save point", "No averaged hits to save.")
            return False
        if s.pending():
            QMessageBox.information(self, "Save point", "Decide on the pending hit first.")
            return False
        cfg = copy.deepcopy(self.session_cfg)
        cur = self.meas.get_save()
        pt = (cfg.save.point_mode, cfg.save.point_node, cfg.save.point_dir)
        cfg.save = cur
        cfg.save.point_mode, cfg.save.point_node, cfg.save.point_dir = pt
        if not cfg.save.folder:
            QMessageBox.warning(self, "Save point", "Choose an output folder first.")
            return False
        folder = point_folder(cfg)
        self.meas.append_log(f"Saving {s.n_included()} averages ({self._point_text(cfg)}) → {folder}")
        try:
            save_point(folder, cfg, s, self.pulses, s.fs, log=self.meas.append_log)
        except Exception as e:
            QMessageBox.critical(self, "Save point", f"{type(e).__name__}: {e}")
            return False
        self.meas.set_phase(f"Saved {self._point_text(cfg)} → {os.path.basename(folder)}")
        if self.worker is not None:
            # still acquiring (roving hammer): loading the run now would hold up the acquisition thread
            self._viewer_pending = (folder, cfg.save.folder)
        else:
            self._viewer_pending = None              # the newest point wins over one queued earlier
            self.viewer.open_folder(folder, parent=cfg.save.folder)
        self.meas.advance_point()
        self.session = None
        self.session_cfg = None
        self.meas.set_session(None)
        self.settings.setValue("last_setup", self.gather_config().to_json())
        return True

    def discard_point(self):
        s = self.session
        if s is not None and s.hits and QMessageBox.question(
                self, "Discard point", f"Forget all {len(s.hits)} hit(s) of this point?") != QMessageBox.Yes:
            return
        self.meas.append_log("Point discarded.")
        if self.worker is not None and self.mode == "measure":
            cfg = self.gather_config()
            seg = self.segment
            self._new_session(cfg)
            self.segment = seg
            self.session.segment = seg
            self._completing = False
            self.worker.new_point(cfg.save.point_node)
        else:
            self.session = None
            self.session_cfg = None
            self.meas.set_session(None)
            self.meas.set_phase("Idle")

    def _simulate_hit(self):
        if self.worker is not None:
            self.worker.simulate_hit()

    # ------------------------------------------------------------------ external trigger test
    def test_pulse(self):
        if self._busy():
            return
        t = self.acq.get_trigger()
        if t.frame_sync or not t.kind or t.kind == "PFI":
            QMessageBox.information(self, "Test pulse", "Choose an AO channel or a DO line (frame sync needs no test).")
            return
        if self.act_sim.isChecked():
            self.statusBar().showMessage(f"(simulation) test pulse on {t.physical}", 5000)
            self.meas.append_log(f"(simulation) test pulse on {t.physical}")
            return
        from ..backends import send_test_pulse
        try:
            msg = send_test_pulse(t, self.acq.get_config().fs)
            self.statusBar().showMessage(msg, 8000)
            self.meas.append_log(msg)
        except Exception as e:
            QMessageBox.critical(self, "Test pulse", f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------ finish
    @Slot(str)
    def _on_failed(self, msg: str):
        self.meas.append_log("ERROR: " + msg.splitlines()[0])
        QMessageBox.critical(self, "Acquisition error", msg)

    @Slot(object)
    def _on_finished(self, summary):
        mode = self.mode
        self.worker = None
        self.mode = None
        self._set_editing_enabled(True)
        self.meas.set_running(False)
        self.acq.probe.set_running(False)
        if summary is not None:
            self.meas.append_log(f"Stopped after {summary.samples / max(summary.fs, 1):.1f} s, {summary.hits} hit(s) detected.")
        if mode == "measure":
            s = self.session
            if s is not None:
                if self._save_after_stop:
                    self.save_current_point(auto=True)
                elif s.complete():
                    self.meas.set_phase(f"Complete ({s.n_included()} averages) — review the hits, then 💾 Save point")
                else:
                    self.meas.set_phase(f"Stopped — {s.n_included()} of {s.acq.n_averages} averages. "
                                        "Continue, Save or Discard.")
        self._save_after_stop = False
        self._completing = False
        if self._viewer_pending:
            self._tab_changed()

    def _set_editing_enabled(self, on: bool):
        self.chan.setEnabled(on)
        self.acq.main_split.widget(0).setEnabled(on)
        self.acq.probe.btn_start.setEnabled(on)
        self.meas.btn_start.setEnabled(on)
        for w in self.meas.save_widgets() + self.meas.point_widgets():
            w.setEnabled(on)
        if on:
            self.meas._pmode_changed()
            self.meas.set_running(False)
        for a in self.menuBar().actions():
            a.setEnabled(on)

    def closeEvent(self, ev):
        if self.worker is not None:
            if QMessageBox.question(self, "Quit", "The acquisition is running. Stop it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(15000)
        if self.session is not None and self.session.n_included() and QMessageBox.question(
                self, "Quit", "The current point has unsaved hits. Quit anyway?") != QMessageBox.Yes:
            ev.ignore()
            return
        self.settings.setValue("last_setup", self.gather_config().to_json())
        ev.accept()
