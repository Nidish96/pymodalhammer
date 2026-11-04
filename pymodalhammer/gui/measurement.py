# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Measurement tab: what to save, the measurement point (hammer node), start/stop, the list of hits
(include, remove, accept or reject any of them), and the current and earlier hits in the time and
frequency domains with the running FRF."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from ..config import DIRECTIONS, POINT_MODES, ChannelConfig, SaveConfig
from ..processing import (ACCEPTED, FLAGGED, PENDING, REJECTED, STATUS_COLORS, Hit, PointSession,
                          force_rolloff)
from .plots import COLOR_A, COLOR_B, EST_COLORS, DualAxisPlot, FRFStack, export_dialog

OVERLAYS = ["All hits (averaged ones in colour)", "Averaged hits only", "Selected / last hit only"]
HIT_COLS = ["#", "Seg.", "Time (s)", "Hammer peak", "Status", "Checks / notes"]
MAX_OVERLAY = 60


def _faint(rgb, alpha=70):
    return pg.mkPen(QColor(*rgb, alpha), width=1)


class MeasurementTab(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    save_requested = Signal()
    discard_requested = Signal()
    decision = Signal(int, bool)             # hit number, accepted
    changed_session = Signal()               # included hits changed (complete? → main window)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Optional[PointSession] = None
        self.channels: List[ChannelConfig] = []
        self._running = False

        # ---------------- save options
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Folder for result files")
        browse = QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse)
        self.basename = QLineEdit()
        self.save_uff = QCheckBox("Frequency domain → UFF 58:")
        self.uff = {k: QCheckBox(k) for k in ("H1", "H2", "Hv", "Coherence")}
        self.uff_auto = QCheckBox("Auto-spectra")
        self.uff_auto.setToolTip("Averaged |X|² of the hammer and |Y|² of every response (UFF 58, function type 2)")
        self.save_tdms = QCheckBox("Hits → TDMS")
        self.save_tdms.setToolTip("Every averaged hit (all channels, + trigger output) → <base>_hits.tdms, one group per "
                                  "hit with status, checks and trigger sample as properties")
        self.save_rej = QCheckBox("include rejected hits")
        self.save_rej.setToolTip("Also store hits that were rejected (marked included = False), so you can "
                                 "re-process them in the Data Viewer. Deleted hits are never saved.")
        self.uff_time = QCheckBox("Hits → UFF 58 (time)")
        self.uff_time.setToolTip("The averaged hits as time responses (UFF 58, function type 1), t = 0 at the trigger")
        self.stream = QCheckBox("Continuous record → TDMS")
        self.stream.setToolTip("Everything acquired, from start to stop (one group per Start/Continue), into "
                               "<folder>/<base>_stream_<time>.tdms — as a safety copy or for your own processing")
        self.save_uff.toggled.connect(lambda on: [c.setEnabled(on) for c in list(self.uff.values()) + [self.uff_auto]])
        self.save_tdms.toggled.connect(self.save_rej.setEnabled)
        g_save = QGroupBox("Save results")
        gl = QGridLayout(g_save)
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Folder"))
        fl.addWidget(self.folder, 3)
        fl.addWidget(browse)
        fl.addSpacing(12)
        fl.addWidget(QLabel("Base name"))
        fl.addWidget(self.basename, 1)
        gl.addLayout(fl, 0, 0)
        ul = QHBoxLayout()
        ul.addWidget(self.save_uff)
        for c in self.uff.values():
            ul.addWidget(c)
        ul.addWidget(self.uff_auto)
        ul.addStretch()
        gl.addLayout(ul, 1, 0)
        tl_ = QHBoxLayout()
        tl_.addWidget(QLabel("Time domain:"))
        for w in (self.save_tdms, self.save_rej, self.uff_time, self.stream):
            tl_.addWidget(w)
        tl_.addStretch()
        gl.addLayout(tl_, 3, 0)
        hint = QLabel("Each saved point gets its own time-stamped folder with setup.json, the hit log (CSV) and, with "
                      "the external trigger, the trigger log. FRFs are saved over Δf … fs/2.")
        hint.setStyleSheet("color: gray")
        hint.setWordWrap(True)
        hint.setMinimumWidth(100)
        gl.addWidget(hint, 4, 0)

        # ---------------- measurement point
        self.pmode = QComboBox()
        self.pmode.addItems(POINT_MODES)
        self.pmode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.pmode.setMinimumContentsLength(20)
        self.pnode = QSpinBox()
        self.pnode.setRange(0, 999999)
        self.pdir = QComboBox()
        self.pdir.addItems(DIRECTIONS)
        self.pinc = QCheckBox("after saving, next node +")
        self.pstep = QSpinBox()
        self.pstep.setRange(-1000, 1000)
        self.pmode.currentIndexChanged.connect(self._pmode_changed)
        g_pt = QGroupBox("Measurement point")
        pl = QHBoxLayout(g_pt)
        pl.addWidget(self.pmode, 2)
        self.pnode_l = QLabel("Hammer node")
        pl.addWidget(self.pnode_l)
        pl.addWidget(self.pnode)
        pl.addWidget(self.pdir)
        pl.addWidget(self.pinc)
        pl.addWidget(self.pstep)
        pl.addStretch()

        # ---------------- run controls
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("QPushButton { background:#1e8449; color:white; font-weight:bold; "
                                     "border-radius:4px; padding:0 18px } QPushButton:disabled { background:#95a5a6 }")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton { background:#c0392b; color:white; font-weight:bold; "
                                    "border-radius:4px; padding:0 18px } QPushButton:disabled { background:#95a5a6 }")
        self.btn_save = QPushButton("💾  Save point")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("QPushButton { background:#2b7bb9; color:white; font-weight:bold; "
                                    "border-radius:4px; padding:0 18px } QPushButton:disabled { background:#95a5a6 }")
        self.btn_discard = QPushButton("✕ Discard point")
        self.btn_discard.setMinimumHeight(40)
        self.btn_discard.setEnabled(False)
        self.btn_discard.setToolTip("Forget every hit of this point and start again")
        self.btn_sim = QPushButton("🔨 Simulate a hit")
        self.btn_sim.setMinimumHeight(40)
        self.btn_sim.setVisible(False)
        self.btn_start.clicked.connect(self.start_requested)
        self.btn_stop.clicked.connect(self.stop_requested)
        self.btn_save.clicked.connect(self.save_requested)
        self.btn_discard.clicked.connect(self.discard_requested)
        self.phase = QLabel("Idle")
        self.phase.setStyleSheet("font-weight: bold")
        self.phase.setWordWrap(True)
        self.phase.setMinimumWidth(50)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m averages")
        self.health = QLabel("")
        self.btn_export = QPushButton("📷  Export plots…")
        self.btn_export.clicked.connect(lambda: export_dialog(self, self.right_tabs.currentWidget(),
                                                              "hammer_measurement.png"))
        run = QHBoxLayout()
        for b in (self.btn_start, self.btn_stop, self.btn_save, self.btn_discard, self.btn_sim):
            run.addWidget(b)
        pv = QVBoxLayout()
        pv.addWidget(self.phase)
        pv.addWidget(self.progress)
        pv.addWidget(self.health)
        run.addLayout(pv, 1)
        run.addWidget(self.btn_export)

        # ---------------- review bar (pending hit)
        self.review = QFrame()
        self.review.setStyleSheet("QFrame { background: rgba(243,156,18,60); border: 1px solid #f39c12; "
                                  "border-radius: 4px }")
        self.review_txt = QLabel()
        self.review_txt.setWordWrap(True)
        self.btn_acc = QPushButton("✔ Accept  (A)")
        self.btn_rej = QPushButton("✘ Reject  (R)")
        self.btn_acc.setStyleSheet("QPushButton { background:#27ae60; color:white; font-weight:bold; padding:6px 14px }")
        self.btn_rej.setStyleSheet("QPushButton { background:#c0392b; color:white; font-weight:bold; padding:6px 14px }")
        self.btn_acc.clicked.connect(lambda: self._decide(True))
        self.btn_rej.clicked.connect(lambda: self._decide(False))
        rl = QHBoxLayout(self.review)
        rl.addWidget(self.review_txt, 1)
        rl.addWidget(self.btn_acc)
        rl.addWidget(self.btn_rej)
        self.review.setVisible(False)
        QShortcut(QKeySequence("A"), self, activated=lambda: self.review.isVisible() and self._decide(True))
        QShortcut(QKeySequence("R"), self, activated=lambda: self.review.isVisible() and self._decide(False))
        QShortcut(QKeySequence.Delete, self, activated=self._delete_selected)

        # ---------------- hit list
        self.table = QTableWidget(0, len(HIT_COLS))
        self.table.setHorizontalHeaderLabels(HIT_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        hh = self.table.horizontalHeader()
        for i in range(len(HIT_COLS) - 1):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(HIT_COLS) - 1, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._redraw)
        self.btn_excl = QPushButton("🗑 Delete selected  (Del)")
        self.btn_excl.setToolTip("Delete the selected hits: they leave the list and the average and are not saved")
        self.btn_incl = QPushButton("✔ Use selected")
        self.btn_incl.setToolTip("Put the selected rejected hits into the average")
        self.btn_excl.clicked.connect(self._delete_selected)
        self.btn_incl.clicked.connect(lambda: self._set_selected(True))
        self.hit_sum = QLabel()
        self.live = DualAxisPlot(downsample=True)
        self.live.setLabel("bottom", "Time", "s")
        self.live.setTitle("Live")
        self.live.setMinimumHeight(110)
        self.lvl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#c0392b", style=Qt.DashLine))
        self.live.addItem(self.lvl_line)
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        hl_ = QLabel("<b>Hits</b> — select one or more to highlight them; Ctrl/Shift for several")
        hl_.setWordWrap(True)
        hl_.setMinimumWidth(100)
        ll.addWidget(hl_)
        ll.addWidget(self.table, 1)
        bl = QHBoxLayout()
        bl.addWidget(self.btn_excl)
        bl.addWidget(self.btn_incl)
        ll.addLayout(bl)
        ll.addWidget(self.hit_sum)
        lsplit = QSplitter(Qt.Vertical)
        lsplit.addWidget(lw)
        lsplit.addWidget(self.live)
        lsplit.setSizes([420, 150])

        # ---------------- right: views
        self.resp = QComboBox()
        self.resp.setStyleSheet(f"QComboBox {{ color: {COLOR_B} }}")
        self.overlay = QComboBox()
        self.overlay.addItems(OVERLAYS)
        self.zoom = QCheckBox("Zoom hammer on the impact")
        self.zoom.setChecked(True)
        self.zoom.setToolTip("Show only the pulse (and the force window) in the hammer plot; the response plot always "
                             "shows the whole record")
        for w in (self.resp, self.overlay):
            w.currentIndexChanged.connect(self._redraw)
        self.zoom.toggled.connect(self._redraw)
        # time
        self.p_f = pg.PlotWidget()
        self.p_r = pg.PlotWidget()
        for p in (self.p_f, self.p_r):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.getAxis("left").enableAutoSIPrefix(False)
        self.p_f.setLabel("bottom", "Time from trigger", "s")
        self.p_r.setLabel("bottom", "Time from trigger", "s")
        tsp = QSplitter(Qt.Vertical)
        tsp.addWidget(self.p_f)
        tsp.addWidget(self.p_r)
        # spectra
        self.p_fs = pg.PlotWidget()
        self.p_fs.setLabel("left", "Hammer", "dB re low-freq. level")
        self.p_rs = pg.PlotWidget()
        self.p_rs.setLogMode(y=True)
        self.p_rs.setLabel("bottom", "Frequency", "Hz")
        for p in (self.p_fs, self.p_rs):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.getAxis("left").enableAutoSIPrefix(False)        # keep 'g' as g, not 'mg' 
        self.p_rs.setXLink(self.p_fs)
        self.f10 = pg.InfiniteLine(angle=90, pen=pg.mkPen("#27ae60", width=1.5), label="−10 dB",
                                   labelOpts={"position": 0.9, "color": "#27ae60"})
        ssp = QSplitter(Qt.Vertical)
        ssp.addWidget(self.p_fs)
        ssp.addWidget(self.p_rs)
        # FRF
        self.est_cb = {k: QCheckBox(k) for k in ("H1", "H2", "Hv")}
        for k, cb in self.est_cb.items():
            cb.setChecked(k == "H1")
            cb.setStyleSheet(f"QCheckBox {{ color: {EST_COLORS[k]}; font-weight: bold }}")
            cb.toggled.connect(self._redraw_frf)
        self.single = QCheckBox("selected hit's own FRF (grey)")
        self.single.setChecked(True)
        self.single.toggled.connect(self._redraw_frf)
        self.navg = QLabel()
        self.frf = FRFStack()
        self.c_single_m = self.frf.p_mag.plot(pen=pg.mkPen((90, 90, 90, 150), width=1))
        self.c_single_p = self.frf.p_ph.plot(pen=pg.mkPen((90, 90, 90, 150), width=1))
        fw = QWidget()
        fl2 = QVBoxLayout(fw)
        fl2.setContentsMargins(0, 0, 0, 0)
        fh = QHBoxLayout()
        for cb in self.est_cb.values():
            fh.addWidget(cb)
        fh.addWidget(self.single)
        fh.addStretch()
        fh.addWidget(self.navg)
        fl2.addLayout(fh)
        fl2.addWidget(self.frf, 1)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(tsp, "Time domain")
        self.right_tabs.addTab(ssp, "Spectra")
        self.right_tabs.addTab(fw, "FRF && coherence")
        self.right_tabs.currentChanged.connect(self._redraw)
        rw = QWidget()
        rl2 = QVBoxLayout(rw)
        rl2.setContentsMargins(0, 0, 0, 0)
        sel = QHBoxLayout()
        sel.addWidget(QLabel("<b>Response</b>"))
        sel.addWidget(self.resp, 1)
        sel.addWidget(QLabel("Show"))
        sel.addWidget(self.overlay, 1)
        sel.addWidget(self.zoom)
        rl2.addLayout(sel)
        rl2.addWidget(self.right_tabs, 1)

        self.plots = QSplitter(Qt.Horizontal)
        self.plots.addWidget(lsplit)
        self.plots.addWidget(rw)
        self.plots.setSizes([480, 900])
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        main = QSplitter(Qt.Vertical)
        main.addWidget(self.plots)
        main.addWidget(self.log)
        main.setSizes([620, 110])
        lay = QVBoxLayout(self)
        lay.addWidget(g_save)
        lay.addWidget(g_pt)
        lay.addLayout(run)
        lay.addWidget(self.review)
        lay.addWidget(main, 1)

    # ------------------------------------------------------------------ save config
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Result folder", self.folder.text() or os.getcwd())
        if d:
            self.folder.setText(d)

    def _pmode_changed(self, *_):
        rov = self.pmode.currentIndex() == 0
        for w in (self.pnode, self.pdir, self.pinc, self.pstep, self.pnode_l):
            w.setEnabled(rov)

    def set_save(self, s: SaveConfig):
        self.folder.setText(s.folder)
        self.basename.setText(s.basename)
        self.save_uff.setChecked(s.save_uff)
        for k, attr in (("H1", "uff_h1"), ("H2", "uff_h2"), ("Hv", "uff_hv"), ("Coherence", "uff_coherence")):
            self.uff[k].setChecked(getattr(s, attr))
            self.uff[k].setEnabled(s.save_uff)
        self.uff_auto.setChecked(s.uff_auto)
        self.uff_auto.setEnabled(s.save_uff)
        self.save_tdms.setChecked(s.save_hits_tdms)
        self.save_rej.setChecked(s.save_rejected)
        self.save_rej.setEnabled(s.save_hits_tdms)
        self.uff_time.setChecked(s.uff_time)
        self.stream.setChecked(s.save_stream_tdms)
        self.pmode.setCurrentText(s.point_mode)
        self.pnode.setValue(s.point_node)
        self.pdir.setCurrentText(s.point_dir)
        self.pinc.setChecked(s.auto_increment)
        self.pstep.setValue(s.increment)
        self._pmode_changed()

    def get_save(self) -> SaveConfig:
        return SaveConfig(folder=self.folder.text().strip(), basename=self.basename.text().strip() or "impact_test",
                          save_uff=self.save_uff.isChecked(), uff_h1=self.uff["H1"].isChecked(),
                          uff_h2=self.uff["H2"].isChecked(), uff_hv=self.uff["Hv"].isChecked(),
                          uff_coherence=self.uff["Coherence"].isChecked(), uff_auto=self.uff_auto.isChecked(),
                          uff_time=self.uff_time.isChecked(), save_hits_tdms=self.save_tdms.isChecked(),
                          save_rejected=self.save_rej.isChecked(), save_stream_tdms=self.stream.isChecked(),
                          point_mode=self.pmode.currentText(), point_node=self.pnode.value(),
                          point_dir=self.pdir.currentText(), auto_increment=self.pinc.isChecked(),
                          increment=self.pstep.value())

    def advance_point(self):
        if self.pmode.currentIndex() == 0 and self.pinc.isChecked():
            self.pnode.setValue(self.pnode.value() + self.pstep.value())

    def save_widgets(self):
        return [self.folder, self.basename, self.save_uff, self.save_tdms, self.save_rej, self.uff_time, self.stream,
                self.uff_auto, *self.uff.values()]

    def point_widgets(self):
        return [self.pmode, self.pnode, self.pdir]

    # ------------------------------------------------------------------ session
    def set_session(self, session: Optional[PointSession], trig_level: float = 0.0):
        """A new (or no) measurement point."""
        self.session = session
        self.review.setVisible(False)
        self.table.setRowCount(0)
        if session is not None:
            self.channels = session.channels
            ref_i = session.ref_i
            items = [f"{c.name}  ({c.node}{c.direction})  [{c.units}]" for i, c in enumerate(self.channels) if i != ref_i]
            cur = self.resp.currentText()
            self.resp.blockSignals(True)
            self.resp.clear()
            self.resp.addItems(items)
            if cur in items:
                self.resp.setCurrentText(cur)
            self.resp.blockSignals(False)
            self.progress.setMaximum(session.acq.n_averages)
            self.frf.set_band(None)
            self._trig_level = trig_level
        self.progress.setValue(0)
        self._update_buttons()
        self._redraw()

    def set_running(self, running: bool):
        self._running = running
        self._update_buttons()

    def _update_buttons(self):
        has = self.session is not None and len(self.session.hits) > 0
        self.btn_start.setEnabled(not self._running)
        self.btn_start.setText("▶  Continue" if has and not self._running else "▶  Start")
        self.btn_stop.setEnabled(self._running)
        self.btn_save.setEnabled(has and self.session.n_included() > 0 and not self.session.pending())
        self.btn_discard.setEnabled(has)

    @Slot(str)
    def append_log(self, s: str):
        self.log.appendPlainText(s)

    def set_phase(self, s: str):
        self.phase.setText(s)

    @Slot(object, int, float, object, int, object)
    def on_scope(self, data, start, fs, peaks, backlog, trig=None):
        if not self.channels or data.shape[0] != len(self.channels):
            return
        n = data.shape[1]
        lag = backlog / max(1, int(0.05 * fs))
        self.health.setText(f"Input backlog: {backlog} samples" + ("  ⚠ PC is falling behind" if lag > 10 else ""))
        ri = self.session.ref_i if self.session else 0
        j = self._resp_index()
        t = (start + np.arange(n)) / fs
        c = self.channels[ri]
        cr = self.channels[j] if j is not None else None
        self.live.set_curves(t, data[ri], c.name, c.units, data[j] if cr else None, cr.name if cr else "",
                             cr.units if cr else "")
        self.lvl_line.setPos(getattr(self, "_trig_level", 0.0))

    def _resp_index(self) -> Optional[int]:
        if self.session is None:
            return None
        resp = [i for i in range(len(self.channels)) if i != self.session.ref_i]
        k = self.resp.currentIndex()
        return resp[k] if 0 <= k < len(resp) else None

    @Slot(object)
    def on_hit(self, hit: Hit):
        if self.session is None:
            return
        self.session.add(hit)
        self._add_row(len(self.session.hits) - 1)
        if hit.status == PENDING:
            why = hit.reason or "manual review"
            self.review_txt.setText(f"<b>Hit {hit.number} waits for your decision</b> — {why}. "
                                    f"Hammer peak {hit.peaks[self.session.ref_i]:.4g} "
                                    f"{self.channels[self.session.ref_i].units}. Acquisition is paused until you decide.")
            self.review.setVisible(True)
        self.table.clearSelection()
        self.table.scrollToBottom()
        self._after_change()

    def _decide(self, accept: bool):
        if self.session is None:
            return
        pend = self.session.pending()
        if not pend:
            self.review.setVisible(False)
            return
        i = pend[0]
        h = self.session.hits[i]
        if accept:
            failed = any(not c.ok for c in h.checks)
            self.session.set_status(i, FLAGGED if failed else ACCEPTED,
                                    (h.reason + " — accepted by you") if h.reason else "accepted by you")
        else:
            self.session.set_status(i, REJECTED, (h.reason + " — rejected by you") if h.reason else "rejected by you")
        self._update_row(i)
        self.review.setVisible(bool(self.session.pending()))
        self.decision.emit(h.number, accept)
        self._after_change()

    def _delete_selected(self):
        if self.session is None:
            return
        rows = [i for i in sorted({i.row() for i in self.table.selectionModel().selectedRows()})
                if self.session.hits[i].status != PENDING]          # a pending hit is decided with the bar
        if not rows:
            return
        gone = self.session.delete(rows)
        for i in reversed(rows):
            self.table.removeRow(i)
        self.table.clearSelection()
        self.append_log("Deleted hit(s) " + ", ".join(str(h.number) for h in gone))
        self._after_change()

    def _set_selected(self, use: bool = True):
        """Put the selected (e.g. rejected) hits into the average."""
        if self.session is None:
            return
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        for i in rows:
            h = self.session.hits[i]
            if h.status == PENDING or h.included:
                continue
            base = h.reason.split(" — ")[0] if h.reason else ""
            failed = any(not c.ok for c in h.checks)
            self.session.set_status(i, FLAGGED if failed else ACCEPTED, (base + " — " if base else "") + "used by you")
            self._update_row(i)
        self._after_change()

    def _after_change(self):
        s = self.session
        n = s.n_included()
        self.progress.setValue(min(n, self.progress.maximum()))
        summ = s.summary()
        self.hit_sum.setText(f"{len(s.hits)} hit(s): " + ", ".join(
            f"<span style='color:rgb{STATUS_COLORS[k]}'>{v} {k}</span>" for k, v in summ.items()))
        self._update_buttons()
        self._redraw()
        self.changed_session.emit()

    def _add_row(self, i: int):
        self.table.insertRow(i)
        for c in range(len(HIT_COLS)):
            self.table.setItem(i, c, QTableWidgetItem())
        self._update_row(i)

    def _update_row(self, i: int):
        h = self.session.hits[i]
        ref = self.session.ref_i
        c = self.channels[ref]
        notes = h.reason or "; ".join(ch.message for ch in h.checks if ch.ok and ch.name == "Double hit")
        vals = [str(h.number), str(h.segment), f"{h.time_s:.2f}", f"{h.peaks[ref]:.4g} {c.units}", h.status, notes]
        col = QColor(*STATUS_COLORS.get(h.status, (127, 127, 127)), 60)
        for k, v in enumerate(vals):
            it = self.table.item(i, k)
            it.setText(v)
            it.setBackground(QBrush(col))
            it.setToolTip("\n".join(f"{ch.name}: {'ok' if ch.ok else 'FAILED'} — {ch.message} ({ch.action})"
                                    for ch in h.checks) or "no checks")

    # ------------------------------------------------------------------ plots
    def _current(self) -> List[int]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        if rows:
            return rows
        return [len(self.session.hits) - 1] if self.session and self.session.hits else []

    def _overlay_hits(self, cur: List[int]) -> List[int]:
        hs = self.session.hits
        m = self.overlay.currentIndex()
        if m == 2:
            return []
        idx = [i for i, h in enumerate(hs) if (m == 0 or h.included) and i not in cur]
        return idx[-MAX_OVERLAY:]

    def _redraw(self, *_):
        for p in (self.p_f, self.p_r, self.p_fs, self.p_rs):
            p.clear()
        self.c_single_m.setData([], [])
        self.c_single_p.setData([], [])
        s = self.session
        if s is None or not s.hits:
            self.frf.clear()
            self.navg.setText("")
            return
        tab = self.right_tabs.currentIndex()
        cur = self._current()
        over = self._overlay_hits(cur)
        ref, j = s.ref_i, self._resp_index()
        n, pre, fs = s.n, s.pre, s.fs
        cf = self.channels[ref]
        cr = self.channels[j] if j is not None else None
        if tab == 0:
            t = (np.arange(n) - pre) / fs
            for i in over:
                h = s.hits[i]
                col = STATUS_COLORS.get(h.status, (127, 127, 127)) if h.included else (150, 150, 150)
                self.p_f.plot(t, h.data[ref], pen=_faint(col))
                if j is not None:
                    self.p_r.plot(t, h.data[j], pen=_faint(col))
            fmax = max([np.max(np.abs(s.hits[i].data[ref])) for i in cur] + [1e-30])
            for i in cur:
                h = s.hits[i]
                self.p_f.plot(t, h.data[ref], pen=pg.mkPen(COLOR_A, width=2))
                if j is not None:
                    self.p_r.plot(t, h.data[j], pen=pg.mkPen(COLOR_B, width=1.5))
            self.p_f.plot(t, s.win.w_force_only * fmax, pen=pg.mkPen("#1e8449", width=1.5, style=Qt.DashLine))
            self.p_f.addItem(pg.InfiniteLine(pos=getattr(self, "_trig_level", 0), angle=0,
                                             pen=pg.mkPen("#c0392b", style=Qt.DotLine)))
            if j is not None and cur:
                rmax = max(np.max(np.abs(s.hits[i].data[j])) for i in cur)
                for sg in (1, -1):
                    self.p_r.plot(t, sg * s.win.w_exp * rmax, pen=pg.mkPen("#8e44ad", width=1.5, style=Qt.DashLine))
            self.p_f.setLabel("left", cf.name, cf.units)
            if cr:
                self.p_r.setLabel("left", cr.name, cr.units)
            self.p_f.setTitle(f"Hammer — bold: hit {', '.join(str(s.hits[i].number) for i in cur[:8])}"
                              + ("…" if len(cur) > 8 else "") + "; green dashed: force window")
            self.p_r.setTitle("Response — purple dashed: exponential window" if np.isfinite(s.win.tau) else "Response")
            if self.zoom.isChecked():
                a = s.acq
                self.p_f.setXRange(-min(pre / fs, 0.005), (a.force_width_ms + a.force_taper_ms) * 1.5e-3, padding=0.02)
            else:
                self.p_f.setXRange(t[0], t[-1], padding=0.01)
            self.p_r.setXRange(t[0], t[-1], padding=0.01)
        elif tab == 1:
            f = s.freqs[1:]

            def fdb(h):
                a = np.abs(h.spectra[ref])
                k = max(1, int(0.02 * a.size))
                return 20 * np.log10(a / (np.max(a[1:k + 1]) + 1e-30) + 1e-30)[1:]

            for i in over:
                h = s.hits[i]
                col = STATUS_COLORS.get(h.status, (127, 127, 127)) if h.included else (150, 150, 150)
                self.p_fs.plot(f, fdb(h), pen=_faint(col))
                if j is not None:
                    self.p_rs.plot(f, np.abs(h.spectra[j, 1:]) + 1e-30, pen=_faint(col))
            for i in cur:
                h = s.hits[i]
                self.p_fs.plot(f, fdb(h), pen=pg.mkPen(COLOR_A, width=2))
                if j is not None:
                    self.p_rs.plot(f, np.abs(h.spectra[j, 1:]) + 1e-30, pen=pg.mkPen(COLOR_B, width=1.5))
            res = s.result()
            if res is not None:
                k = max(1, int(0.02 * res.Gxx.size))
                avg = 10 * np.log10(res.Gxx / (np.max(res.Gxx[1:k + 1]) + 1e-300) + 1e-300)
                self.p_fs.plot(f, avg[1:], pen=pg.mkPen("k", width=2), name="average")
                f10 = force_rolloff(res.freqs, np.sqrt(res.Gxx), -10)
                self.f10.setPos(f10)
                self.p_fs.addItem(self.f10)
                if j is not None:
                    jj = res.resp_i.index(j)
                    self.p_rs.plot(f, np.sqrt(res.Gyy[jj, 1:]) + 1e-30, pen=pg.mkPen("k", width=2))
            for db in (-10, -20):
                self.p_fs.addItem(pg.InfiniteLine(pos=db, angle=0, pen=pg.mkPen("#7f8c8d", style=Qt.DashLine)))
            self.p_fs.setYRange(-50, 5)
            self.p_fs.setTitle("Hammer force spectrum, windowed (faint: hits, bold blue: selected, black: average)")
            self.p_rs.setTitle("Response |Y(f)| (black: RMS average of the averaged hits)")
            if cr:
                self.p_rs.setLabel("left", f"|{cr.name}|", f"{cr.units}·s")
        else:
            self._redraw_frf()
        for p in (self.p_f, self.p_r, self.p_fs, self.p_rs):   # cheap redraws: the acquisition thread shares the GIL
            for it in p.listDataItems():
                it.setDownsampling(auto=True, method="peak")
                it.setClipToView(True)

    def _redraw_frf(self, *_):
        s = self.session
        if s is None:
            return
        res = s.result()
        j = self._resp_index()
        cr = self.channels[j] if j is not None else None
        cf = self.channels[s.ref_i]
        if res is None or j is None:
            self.frf.clear()
            self.navg.setText("no averaged hits yet")
        else:
            jj = res.resp_i.index(j)
            curves = {k: getattr(res, k)[jj][1:] if self.est_cb[k].isChecked() else None for k in ("H1", "H2", "Hv")}
            self.frf.set_frf(res.freqs[1:], curves, res.coherence[jj][1:], f"|H|  [{cr.units}/{cf.units}]")
            self.navg.setText(f"{res.n_avg} average(s)")
        cur = self._current()
        if self.single.isChecked() and cur and j is not None:
            h = s.hits[cur[-1]]
            H = h.spectra[j] / (h.spectra[s.ref_i] + 1e-300)
            self.c_single_m.setData(s.freqs[1:], np.abs(H[1:]) + 1e-30)
            self.c_single_p.setData(s.freqs[1:], np.degrees(np.angle(H[1:])))
        else:
            self.c_single_m.setData([], [])
            self.c_single_p.setData([], [])
