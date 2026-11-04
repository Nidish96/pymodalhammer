# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Acquisition Setup tab: record, trigger, windows, averaging and acceptance, hit checks (reject /
fail behaviour) and the external trigger output. Right: a summary, a preview of the record with
its windows (over the last real hit when there is one), the trigger pulse, and Probe mode."""

from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget)

from ..config import (ACCEPT_MODES, CHECK_ACTIONS, DONE_ACTIONS, FORCE_WINDOWS, FRAME_SYNC, NOCHECK,
                      RESP_WINDOWS, TRIGGER_MODE_TIPS, TRIGGER_MODES, TRIGGER_POLARITY, TRIGGER_SLOPES,
                      TRIGGER_SYNC, TRIGGER_TIMING, AcquisitionConfig, TriggerConfig, output_kind)
from ..processing import Windows, added_damping
from .probe_panel import ProbePanel

COMMON_RATES = ["1024", "2048", "2560", "5120", "10240", "12800", "25600", "51200"]
BLOCK_SIZES = [str(2 ** k) for k in range(9, 19)]
REF_ITEM = "(the hammer / reference channel)"
ACTION_TIPS = {
    "Reject automatically": "The hit is not used; acquisition re-arms at once.",
    "Ask me": "Acquisition waits; you accept or reject the hit on the Measurement tab.",
    "Accept, but flag it": "The hit is averaged but marked (orange) so you can remove it later.",
    "Don't check": "This check is switched off.",
}


def _dspin(lo, hi, dec, suffix="", step=None):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    if suffix:
        w.setSuffix(" " + suffix)
    if step:
        w.setSingleStep(step)
    return w


def _ispin(lo, hi, suffix=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    if suffix:
        w.setSuffix(" " + suffix)
    return w


def _action_combo():
    c = QComboBox()
    c.addItems(CHECK_ACTIONS)
    for i, a in enumerate(CHECK_ACTIONS):
        c.setItemData(i, ACTION_TIPS[a], 3)
    return c


class AcquisitionSetup(QWidget):
    changed = Signal()
    test_pulse_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._units = {}
        self._ref_name = ""
        self._example = None            # last real hit (data, pre, fs, ref_i, resp_i, names)

        # ---------------- record
        self.fs = QComboBox()
        self.fs.setEditable(True)
        self.fs.addItems(COMMON_RATES)
        self.fs.setToolTip("Delta-sigma modules (NI 9234 …) only allow 51.2 kHz / n; DAQmx coerces the rate and "
                           "the run uses the rate it really gets (written to the log and the files).")
        self.n = QComboBox()
        self.n.setEditable(True)
        self.n.addItems(BLOCK_SIZES)
        self.n_info = QLabel()
        self.pre = _dspin(0, 49, 1, "% of record", 1)
        self.pre.setToolTip("Part of the record before the trigger point. Keep it short but long enough to see "
                            "the start of the pulse (and to estimate the sensor offsets).")
        self.pre_info = QLabel()
        self.chunk = _dspin(5, 1000, 0, "ms", 10)
        self.chunk.setToolTip("Data are read in pieces of this length. Shorter = faster detection (and a faster "
                              "external trigger pulse), more CPU load. Over USB keep it ≥ 20 ms.")
        self.settle = _dspin(0, 120, 1, "s", 0.5)
        self.settle.setToolTip("Triggers are ignored for this time after the start: IEPE sensors and AC coupling "
                               "need to settle.")
        g_rec = QGroupBox("Record")
        f = QFormLayout(g_rec)
        f.addRow("Sampling rate (Hz)", self.fs)
        rl = QHBoxLayout()
        rl.addWidget(self.n, 1)
        rl.addWidget(self.n_info)
        f.addRow("Record length (samples)", rl)
        pl = QHBoxLayout()
        pl.addWidget(self.pre, 1)
        pl.addWidget(self.pre_info)
        f.addRow("Pre-trigger", pl)
        f.addRow("Read size", self.chunk)
        f.addRow("IEPE settling time", self.settle)

        # ---------------- trigger
        self.trig_ch = QComboBox()
        self.trig_ch.setToolTip("Normally the hammer. Any enabled channel can be used.")
        self.trig_level = _dspin(0.001, 1e9, 3, "", 5)
        self.trig_level.setToolTip("Trigger when the signal crosses this level (in the channel's units). Set it well "
                                   "above the noise and well below a light tap; Probe mode suggests a value.")
        self.trig_unit = QLabel()
        self.slope = QComboBox()
        self.slope.addItems(TRIGGER_SLOPES)
        self.rearm = _dspin(0, 99, 0, "% of level", 5)
        self.rearm.setToolTip("After a record the trigger re-arms only once the signal has dropped below this "
                              "fraction of the level (hysteresis), so a ringing hammer cannot re-trigger.")
        self.holdoff = _dspin(0, 60, 2, "s", 0.1)
        self.holdoff.setToolTip("Extra dead time after each record before the next hit is accepted.")
        self.offset = QCheckBox("Remove each channel's offset (pre-trigger mean)")
        self.offset.setToolTip("Subtracts the mean of the pre-trigger part of every channel (sensor offset, slow "
                               "AC-coupling drift) before the windows and the FFT.")
        g_trg = QGroupBox("Trigger (hit detection)")
        f = QFormLayout(g_trg)
        f.addRow("Channel", self.trig_ch)
        tl = QHBoxLayout()
        tl.addWidget(self.trig_level, 1)
        tl.addWidget(self.trig_unit)
        f.addRow("Level", tl)
        f.addRow("Slope", self.slope)
        f.addRow("Re-arm below", self.rearm)
        f.addRow("Hold-off", self.holdoff)
        f.addRow("", self.offset)

        # ---------------- windows
        self.fwin = QComboBox()
        self.fwin.addItems(FORCE_WINDOWS)
        self.fwin.setToolTip("Force window: 1 from the record start to the end of the pulse, then 0. Removes the "
                             "noise the hammer channel picks up after the impact.")
        self.fwidth = _dspin(0.1, 10000, 2, "ms after trigger", 1)
        self.ftaper = _dspin(0, 1000, 2, "ms cosine taper", 0.5)
        self.rwin = QComboBox()
        self.rwin.addItems(RESP_WINDOWS)
        self.rwin.setToolTip("Exponential window: forces the responses to decay within the record (no leakage), "
                             "at the price of added damping (shown on the right). It is applied to the force "
                             "too, so the FRF stays consistent.")
        self.exp_end = _dspin(0.01, 100, 2, "% at record end", 1)
        self.win_info = QLabel()
        self.win_info.setWordWrap(True)
        self.win_info.setStyleSheet("color: gray")
        g_win = QGroupBox("Windows")
        f = QFormLayout(g_win)
        f.addRow("Hammer", self.fwin)
        f.addRow("  width", self.fwidth)
        f.addRow("  taper", self.ftaper)
        f.addRow("Responses (+ force)", self.rwin)
        f.addRow("  end value", self.exp_end)
        f.addRow(self.win_info)

        # ---------------- averaging and acceptance
        self.navg = _ispin(1, 10000, "averages")
        self.accept = QComboBox()
        self.accept.addItems(ACCEPT_MODES)
        self.accept.setItemData(0, "Each hit is accepted, flagged, rejected or held for your decision by the checks "
                                   "below. You can still remove any hit later.", 3)
        self.accept.setItemData(1, "Every hit waits for Accept / Reject on the Measurement tab (the check results are "
                                   "shown as advice). Acquisition waits meanwhile.", 3)
        self.done = QComboBox()
        self.done.addItems(DONE_ACTIONS)
        for cb in (self.accept, self.done, self.slope):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(14)
        g_avg = QGroupBox("Averaging && acceptance")
        f = QFormLayout(g_avg)
        f.addRow("Number of averages", self.navg)
        f.addRow("Accept hits", self.accept)
        f.addRow("When complete", self.done)

        # ---------------- checks (reject / fail behaviour)
        self.ov_pct = _dspin(50, 100, 1, "% of range", 1)
        self.ov_act = _action_combo()
        self.dbl_pct = _dspin(1, 100, 1, "% of 1st peak", 1)
        self.dbl_act = _action_combo()
        self.fmin = _dspin(0, 1e9, 2, "", 10)
        self.fmax = _dspin(0, 1e9, 2, "", 10)
        self.f_act = _action_combo()
        self.dec_pct = _dspin(0.1, 100, 1, "% of its peak", 1)
        self.dec_act = _action_combo()
        g_chk = QGroupBox("Hit checks — what to do with a bad hit")
        g = QGridLayout(g_chk)
        g.setColumnStretch(1, 1)
        rows = [("Overload", "any channel ≥", self.ov_pct, self.ov_act,
                 "Clipped signals give wrong FRFs. Lower the hit force or raise the channel's range."),
                ("Double hit", "later force peak >", self.dbl_pct, self.dbl_act,
                 "The hammer bounced: after the main pulse has ended the force rose again above this level."),
                ("Force level", None, None, self.f_act,
                 "The peak hammer force must lie within the range (consistent hits average better)."),
                ("Not decayed", "response at end >", self.dec_pct, self.dec_act,
                 "A response still ringing at the record end (RMS of the last 5 %) leaks. Use a longer record "
                 "or a stronger exponential window.")]
        for r, (name, txt, w, act, tip) in enumerate(rows):
            lbl = QLabel(f"<b>{name}</b>")
            lbl.setToolTip(tip)
            g.addWidget(lbl, 2 * r, 0)
            if w is not None:
                h = QHBoxLayout()
                h.addWidget(QLabel(txt))
                h.addWidget(w, 1)
                g.addLayout(h, 2 * r, 1)
            else:
                h = QHBoxLayout()
                h.addWidget(self.fmin, 1)
                h.addWidget(QLabel("to"))
                h.addWidget(self.fmax, 1)
                self.f_unit = QLabel()
                h.addWidget(self.f_unit)
                g.addLayout(h, 2 * r, 1)
            g.addWidget(act, 2 * r + 1, 1)
            ql = QLabel("→")
            ql.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            g.addWidget(ql, 2 * r + 1, 0)

        # ---------------- external trigger
        self.trg = QGroupBox("External trigger (e.g. high-speed camera)")
        self.trg.setCheckable(True)
        self.trg.setChecked(False)
        self.trg.setToolTip("Send a pulse on an AO channel or a DO line when something happens, or route the "
                            "sample clock to a PFI pin (frame sync).")
        self.trg_chan = QComboBox()
        self.trg_chan.setEditable(True)
        self.trg_chan.setToolTip("An AO channel (…/aoN) or a digital line (…/portN/lineM); hardware-timed DO needs a "
                                 "correlated DIO module such as the NI 9401. Frame sync: a PFI terminal.")
        self.trg_kind = QLabel()
        self.trg_mode = QComboBox()
        self.trg_mode.addItems(TRIGGER_MODES)
        for i, tip in enumerate(TRIGGER_MODE_TIPS):
            self.trg_mode.setItemData(i, tip, 3)
        self.trg_mode.view().setMinimumWidth(430)
        self.trg_timing = QComboBox()
        self.trg_timing.addItems(TRIGGER_TIMING)
        self.trg_timing.setItemData(0, "A continuous output task on the AI sample clock. The pulse is written into the "
                                       "output queue, so it leaves (queue depth × read size) after the decision, "
                                       "at an exactly known sample index.", 3)
        self.trg_timing.setItemData(1, "The line is switched by a software write as soon as the decision is made "
                                       "(a few ms over USB); the logged sample index is approximate. Needs a module "
                                       "that allows on-demand output: DO lines and most AO modules, but NOT "
                                       "delta-sigma AO such as the NI 9260.", 3)
        self.trg_sync = QComboBox()
        self.trg_sync.addItems(TRIGGER_SYNC)
        self.trg_sync.setItemData(0, "The output runs on its own clock at the AI rate and starts on the AI start "
                                     "trigger (as 'Shared start trigger' in the shaker app). Works with every module, "
                                     "including delta-sigma ones such as the NI 9234 + NI 9260.", 3)
        self.trg_sync.setItemData(1, "The output is clocked by the AI sample clock itself (exact lockstep). Not every "
                                     "module combination allows it; if DAQmx refuses, the run falls back to the "
                                     "first option and says so in the log.", 3)
        self.trg_width = _ispin(1, 1_000_000, "samples")
        self.trg_width_t = QLabel()
        self.trg_level_v = _dspin(0.1, 10, 2, "V", 0.5)
        self.trg_level_v.setToolTip("Active level of an analog trigger (TTL inputs read ≥ 2 V as high)")
        self.trg_pol = QComboBox()
        self.trg_pol.addItems(TRIGGER_POLARITY)
        self.trg_queue = _dspin(50, 10000, 0, "ms ahead", 100)
        self.trg_queue.setToolTip("Hardware-timed output: how much trigger output is queued ahead of the acquisition. "
                                  "It is the slack the PC has before the output runs dry (DAQmx error -200621), and "
                                  "also the delay of every hit-based pulse. 500 ms suits most PCs; use 1000–2000 ms "
                                  "if you see underflow errors.")
        self.trg_probe = QCheckBox("Also send pulses in Probe mode")
        self.btn_test = QPushButton("⚡ Send a test pulse now")
        self.btn_test.setToolTip("One pulse on the trigger line (when nothing is running), to check the wiring")
        self.btn_test.clicked.connect(self.test_pulse_requested)
        for cb in (self.trg_chan, self.trg_mode, self.trg_pol, self.trg_timing, self.trg_sync):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
        f = QFormLayout(self.trg)
        tc = QHBoxLayout()
        tc.addWidget(self.trg_chan, 1)
        tc.addWidget(self.trg_kind)
        f.addRow("Channel", tc)
        f.addRow("Send pulse", self.trg_mode)
        f.addRow("Timing", self.trg_timing)
        f.addRow("Synchronisation", self.trg_sync)
        tw = QHBoxLayout()
        tw.addWidget(self.trg_width, 1)
        tw.addWidget(self.trg_width_t)
        f.addRow("Pulse width", tw)
        f.addRow("Level (AO only)", self.trg_level_v)
        f.addRow("Polarity", self.trg_pol)
        f.addRow("Output queue", self.trg_queue)
        f.addRow("", self.trg_probe)
        f.addRow("", self.btn_test)

        # ---------------- right: summary, previews, probe
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setStyleSheet("QLabel { background: palette(alternate-base); padding: 8px; }")
        self.p_rec = pg.PlotWidget(title="Record and windows")
        self.p_rec.setLabel("bottom", "Time from trigger", "s")
        self.p_rec.setLabel("left", "Normalised")
        self.p_rec.showGrid(x=True, y=True, alpha=0.3)
        self.p_rec.getAxis("bottom").enableAutoSIPrefix(False)
        self.p_rec.addLegend(offset=(-10, 10))
        self.c_force = self.p_rec.plot(pen=pg.mkPen("#2b7bb9", width=1), name="hammer (last hit)")
        self.c_resp = self.p_rec.plot(pen=pg.mkPen((211, 84, 0, 160), width=1), name="response (last hit)")
        self.c_wf = self.p_rec.plot(pen=pg.mkPen("#1e8449", width=2), name="force window")
        self.c_we = self.p_rec.plot(pen=pg.mkPen("#8e44ad", width=2, style=Qt.DashLine), name="exponential window")
        self.pre_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(127, 140, 141, 40), pen=pg.mkPen(None))
        self.p_rec.addItem(self.pre_region)
        self.trig_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#c0392b", style=Qt.DashLine))
        self.p_rec.addItem(self.trig_line)
        self.p_trig = pg.PlotWidget(title="Trigger pulse (one event)")
        self.p_trig.showGrid(x=True, y=True, alpha=0.3)
        for ax in ("bottom", "left"):
            self.p_trig.getAxis(ax).enableAutoSIPrefix(False)
        self.c_trig = self.p_trig.plot(pen=pg.mkPen("#8e44ad", width=2), stepMode="center")
        self.c_trig_pts = self.p_trig.plot(pen=None, symbol="o", symbolSize=5, symbolBrush="#8e44ad", symbolPen=None)
        self.p_trig.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#7f8c8d", style=Qt.DashLine)))
        self.p_trig.setVisible(False)
        self.zoom = QCheckBox("Zoom on the pulse")
        self.zoom.setChecked(False)
        self.zoom.toggled.connect(self._draw_preview)

        pv = QWidget()
        pvl = QVBoxLayout(pv)
        pvl.setContentsMargins(0, 0, 0, 0)
        pvl.addWidget(self.summary)
        zl = QHBoxLayout()
        zl.addStretch()
        zl.addWidget(self.zoom)
        pvl.addLayout(zl)
        sp = QSplitter(Qt.Vertical)
        for p in (self.p_rec, self.p_trig):
            p.setMinimumHeight(120)
            sp.addWidget(p)
        sp.setSizes([420, 180])
        pvl.addWidget(sp, 1)
        self.probe = ProbePanel()
        self.right = QTabWidget()
        self.right.addTab(pv, "Summary && windows")
        self.right.addTab(self.probe, "🔨 Probe mode (check levels)")

        left = QVBoxLayout()
        for gb in (g_rec, g_trg, g_win, g_avg, g_chk, self.trg):
            left.addWidget(gb)
        left.addStretch()
        lw = QWidget()
        lw.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidget(lw)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(400)
        self.main_split = QSplitter(Qt.Horizontal)
        self.main_split.addWidget(scroll)
        self.main_split.addWidget(self.right)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setSizes([480, 900])
        self.main_split.setChildrenCollapsible(False)
        lay = QVBoxLayout(self)
        lay.addWidget(self.main_split)

        for w in (self.pre, self.chunk, self.settle, self.trig_level, self.rearm, self.holdoff, self.fwidth,
                  self.ftaper, self.exp_end, self.navg, self.ov_pct, self.dbl_pct, self.fmin, self.fmax,
                  self.dec_pct, self.trg_width, self.trg_level_v, self.trg_queue):
            w.valueChanged.connect(self._on_change)
        for w in (self.fs, self.n, self.trig_ch, self.slope, self.fwin, self.rwin, self.accept, self.done,
                  self.ov_act, self.dbl_act, self.f_act, self.dec_act, self.trg_chan, self.trg_timing, self.trg_pol,
                  self.trg_sync):
            w.currentTextChanged.connect(self._on_change)
        for w in (self.offset, self.trg, self.trg_probe):
            w.toggled.connect(self._on_change)
        self.trg_mode.currentTextChanged.connect(self._mode_changed)
        self.probe.suggest_trigger.connect(self._apply_trigger_suggestion)

    # ------------------------------------------------------------------ channel names / outputs
    def set_channel_names(self, chans, ref_name: str = ""):
        """chans: [(name, units, range)] of the enabled channels."""
        self._units = {n: u for n, u, _ in chans}
        self._ranges = {n: r for n, _, r in chans}
        self._ref_name = ref_name
        cur = self.trig_ch.currentData() if self.trig_ch.count() else getattr(self, "_pending_trig", "")
        self.trig_ch.blockSignals(True)
        self.trig_ch.clear()
        self.trig_ch.addItem(REF_ITEM + (f" — {ref_name}" if ref_name else ""), "")
        for n, u, _ in chans:
            self.trig_ch.addItem(f"{n}  [{u}]", n)
        i = self.trig_ch.findData(cur)
        self.trig_ch.setCurrentIndex(max(i, 0))
        self.trig_ch.blockSignals(False)
        self._on_change()

    def set_output_names(self, ao: List[str], do: List[str], pfi: List[str] = ()):
        self._names = {"pulse": list(ao) + list(do), "pfi": list(pfi)}
        self._fill_trigger_names()

    def _fill_trigger_names(self):
        names = getattr(self, "_names", {"pulse": [], "pfi": []})
        frame = self.trg_mode.currentText() == FRAME_SYNC
        cur = self.trg_chan.currentText().strip()
        memory = self.__dict__.setdefault("_trig_memory", {})
        if output_kind(cur):
            memory["pfi" if output_kind(cur) == "PFI" else "pulse"] = cur
        want = "pfi" if frame else "pulse"
        self.trg_chan.blockSignals(True)
        self.trg_chan.clear()
        self.trg_chan.addItems(names[want])
        self.trg_chan.setCurrentText(memory.get(want) or (names[want][0] if names[want] else ""))
        self.trg_chan.blockSignals(False)

    def _mode_changed(self, *_):
        if not getattr(self, "_loading", False):
            self._fill_trigger_names()
        self._on_change()

    def trig_name(self) -> str:
        return self.trig_ch.currentData() or ""

    def trig_units(self) -> str:
        return self._units.get(self.trig_name() or self._ref_name, "")

    # ------------------------------------------------------------------ data
    def set_config(self, a: AcquisitionConfig):
        self._loading = True
        self.fs.setCurrentText(f"{a.fs:g}")
        self.n.setCurrentText(str(a.block_size))
        self.pre.setValue(a.pretrigger_pct)
        self.chunk.setValue(a.chunk_ms)
        self.settle.setValue(a.settle_s)
        self._pending_trig = a.trig_channel
        i = self.trig_ch.findData(a.trig_channel)
        self.trig_ch.setCurrentIndex(max(i, 0))
        self.trig_level.setValue(a.trig_level)
        self.slope.setCurrentText(a.trig_slope)
        self.rearm.setValue(a.rearm_pct)
        self.holdoff.setValue(a.holdoff_s)
        self.offset.setChecked(a.remove_offset)
        self.fwin.setCurrentText(a.force_window)
        self.fwidth.setValue(a.force_width_ms)
        self.ftaper.setValue(a.force_taper_ms)
        self.rwin.setCurrentText(a.resp_window)
        self.exp_end.setValue(a.exp_end_pct)
        self.navg.setValue(a.n_averages)
        self.accept.setCurrentText(a.accept_mode)
        self.done.setCurrentText(a.when_done)
        self.ov_pct.setValue(a.overload_pct)
        self.ov_act.setCurrentText(a.overload_action)
        self.dbl_pct.setValue(a.double_hit_pct)
        self.dbl_act.setCurrentText(a.double_hit_action)
        self.fmin.setValue(a.force_min)
        self.fmax.setValue(a.force_max)
        self.f_act.setCurrentText(a.force_action)
        self.dec_pct.setValue(a.decay_pct)
        self.dec_act.setCurrentText(a.decay_action)
        self.trg_queue.setValue(a.out_queue_ms)
        self.probe.set_suggest_params(a.probe_headroom, a.probe_trig_pct)
        self._loading = False
        self._on_change()

    def get_config(self) -> AcquisitionConfig:
        def num(txt, typ, default):
            try:
                return typ(float(txt))
            except ValueError:
                return default
        head, tpct = self.probe.suggest_params()
        return AcquisitionConfig(
            fs=num(self.fs.currentText(), float, 10240.0), block_size=num(self.n.currentText(), int, 8192),
            pretrigger_pct=self.pre.value(), trig_channel=self.trig_name(), trig_level=self.trig_level.value(),
            trig_slope=self.slope.currentText(), rearm_pct=self.rearm.value(), holdoff_s=self.holdoff.value(),
            settle_s=self.settle.value(), remove_offset=self.offset.isChecked(), chunk_ms=self.chunk.value(),
            out_queue_ms=self.trg_queue.value(), force_window=self.fwin.currentText(),
            force_width_ms=self.fwidth.value(), force_taper_ms=self.ftaper.value(), resp_window=self.rwin.currentText(),
            exp_end_pct=self.exp_end.value(), n_averages=self.navg.value(), accept_mode=self.accept.currentText(),
            when_done=self.done.currentText(), overload_pct=self.ov_pct.value(), overload_action=self.ov_act.currentText(),
            double_hit_pct=self.dbl_pct.value(), double_hit_action=self.dbl_act.currentText(),
            force_min=self.fmin.value(), force_max=self.fmax.value(), force_action=self.f_act.currentText(),
            decay_pct=self.dec_pct.value(), decay_action=self.dec_act.currentText(),
            probe_headroom=head, probe_trig_pct=tpct)

    def set_trigger(self, t: TriggerConfig):
        self._loading = True
        self.trg.setChecked(t.enabled)
        self.trg_mode.setCurrentIndex(t.mode_index())
        self._fill_trigger_names()
        self.trg_chan.setCurrentText(t.physical)
        self.trg_timing.setCurrentText(t.timing)
        self.trg_sync.setCurrentText(t.sync)
        self.trg_width.setValue(t.pulse_samples)
        self.trg_level_v.setValue(t.level_v)
        self.trg_pol.setCurrentText(t.polarity)
        self.trg_probe.setChecked(t.in_probe)
        self._loading = False
        self._on_change()

    def get_trigger(self) -> TriggerConfig:
        return TriggerConfig(enabled=self.trg.isChecked(), physical=self.trg_chan.currentText().strip(),
                             mode=TRIGGER_MODES[max(0, self.trg_mode.currentIndex())],
                             timing=self.trg_timing.currentText(), pulse_samples=self.trg_width.value(),
                             level_v=self.trg_level_v.value(), polarity=self.trg_pol.currentText(),
                             in_probe=self.trg_probe.isChecked(), sync=self.trg_sync.currentText())

    def _apply_trigger_suggestion(self, level: float):
        self.right.setCurrentIndex(1)
        self.trig_level.setValue(level)

    def set_example_hit(self, data: np.ndarray, pre: int, fs: float, ref_i: int, resp_i: int):
        """Show a real hit behind the window preview (from Probe mode or the Measurement tab)."""
        self._example = (data, pre, fs, ref_i, resp_i)
        self._draw_preview()

    # ------------------------------------------------------------------ summary + preview
    def _on_change(self, *_):
        if getattr(self, "_loading", False):
            return
        a = self.get_config()
        u = self.trig_units()
        self.trig_level.setSuffix(f" {u}" if u else "")
        self.trig_unit.setText("")
        fu = self._units.get(self._ref_name, "")
        self.fmin.setSuffix(f" {fu}")
        self.fmax.setSuffix(f" {fu}")
        self.f_unit.setText("")
        self.fmin.setEnabled(self.f_act.currentText() != NOCHECK)
        self.fmax.setEnabled(self.f_act.currentText() != NOCHECK)
        for w, act in ((self.ov_pct, self.ov_act), (self.dbl_pct, self.dbl_act), (self.dec_pct, self.dec_act)):
            w.setEnabled(act.currentText() != NOCHECK)
        fw = self.fwin.currentIndex() == 0
        self.fwidth.setEnabled(fw)
        self.ftaper.setEnabled(fw)
        self.exp_end.setEnabled(self.rwin.currentIndex() == 0)
        if a.fs > 0 and a.block_size > 0:
            self.n_info.setText(f"= {a.record_s:.4g} s, Δf = {a.df:.4g} Hz")
            self.pre_info.setText(f"= {a.pre_samples} samples, {a.pre_samples / a.fs * 1000:.1f} ms")
        win = Windows.make(a, a.block_size, a.fs, a.pre_samples) if a.fs > 0 and a.block_size >= 64 else None
        if win is not None and np.isfinite(win.tau):
            fr = np.array([10, 50, 100, 500, 1000])
            z = added_damping(win.tau, fr) * 100
            self.win_info.setText(f"τ = {win.tau * 1000:.1f} ms. Added damping ζ_w = 1/(2π f τ): "
                                  + ", ".join(f"{p:.2g} % at {f:g} Hz" for f, p in zip(fr, z) if f < a.fs / 2)
                                  + ". Subtract it from identified damping ratios.")
        else:
            self.win_info.setText("No exponential window: responses must decay inside the record.")
        trig_txt = self._trigger_summary(a)
        tname = self.trig_name() or self._ref_name or "?"
        a_txt = ("<b>automatically</b> by the checks" if not a.manual else "<b>by you</b> (manual review of every hit)")
        self.summary.setText(
            f"<b>Record</b>: {a.block_size} samples at {a.fs:g} Hz = <b>{a.record_s:.4g} s</b>, "
            f"Δf = <b>{a.df:.4g} Hz</b>, 0 … {a.fs / 2:g} Hz; pre-trigger {a.pre_samples} samples "
            f"({a.pre_samples / max(a.fs, 1) * 1000:.1f} ms)."
            f"<br><b>Trigger</b>: {tname} {a.trig_slope.split(' (')[0].lower()} through "
            f"{a.trig_level:g} {u}, re-armed below {a.rearm_pct:g} % of it"
            + (f", hold-off {a.holdoff_s:g} s" if a.holdoff_s else "")
            + f"; hits are ignored for the first {a.settle_s:g} s (sensor settling)."
            f"<br><b>Averaging</b>: {a.n_averages} hits, accepted {a_txt}; then: {a.when_done[0].lower() + a.when_done[1:]}."
            f"<br><b>Checks</b>: " + "; ".join(
                f"{n} → {act.lower()}" for n, act in (("overload", a.overload_action), ("double hit", a.double_hit_action),
                                                     ("force level", a.force_action), ("decay", a.decay_action)))
            + "." + trig_txt)
        self._draw_preview()
        self._draw_trigger(a)
        self.probe.set_trigger_info(a.trig_level, u)
        self.changed.emit()

    def _draw_preview(self, *_):
        a = self.get_config()
        if a.fs <= 0 or a.block_size < 64:
            return
        n, pre, fs = a.block_size, a.pre_samples, a.fs
        ex = self._example
        if ex is not None and ex[0].shape[1] == n:
            data, pre_e, fs_e, ref_i, resp_i = ex
            t = (np.arange(n) - pre_e) / fs_e
            f = data[ref_i] if ref_i >= 0 else np.zeros(n)
            self.c_force.setData(t, f / (np.max(np.abs(f)) + 1e-30))
            r = data[resp_i] if 0 <= resp_i < data.shape[0] else np.zeros(n)
            self.c_resp.setData(t, r / (np.max(np.abs(r)) + 1e-30))
            self.p_rec.setTitle("Record and windows — over the last hit (normalised)")
        else:
            self.c_force.setData([], [])
            self.c_resp.setData([], [])
            self.p_rec.setTitle("Record and windows — no hit yet: use Probe mode or measure to see a real one")
        win = Windows.make(a, n, fs, pre)
        t = (np.arange(n) - pre) / fs
        self.c_wf.setData(t, win.w_force_only)
        self.c_we.setData(t, win.w_exp)
        self.pre_region.setRegion((t[0], 0))
        if self.zoom.isChecked():
            end = (a.force_width_ms + a.force_taper_ms) * 1e-3
            self.p_rec.setXRange(t[0], max(end * 1.5, 0.005), padding=0.02)
        else:
            self.p_rec.setXRange(t[0], t[-1], padding=0.01)
        self.p_rec.setYRange(-1.05, 1.1)

    def _draw_trigger(self, a: AcquisitionConfig):
        t = self.get_trigger()
        self.p_trig.setVisible(t.enabled)
        if not t.enabled or a.fs <= 0:
            return
        dt_us = 1e6 / a.fs
        low_active = t.polarity.startswith("Active low")
        if t.frame_sync:
            n, w = 8, 0.2
            edges, ys = [-1.0], []
            for k in range(n):
                edges += [k, k + w]
                ys += [0.0, 1.0]
            edges.append(n)
            ys.append(0.0)
            y = np.array(ys)
            self.c_trig.setData(np.array(edges), 1.0 - y if low_active else y)
            self.c_trig_pts.setData([], [])
            self.p_trig.setTitle(f"Frame clock: one pulse per sample (every {dt_us:,.1f} µs), width set by hardware")
            self.p_trig.setLabel("left", "Logic level")
            self.p_trig.setLabel("bottom", "Sample index from start of acquisition")
            return
        hi = t.level_v if t.kind == "AO" else 1.0
        active, idle = (0.0, hi) if low_active else (hi, 0.0)
        w = max(1, t.pulse_samples)
        pre, post = max(4, w // 2), max(8, w)
        k = np.arange(-pre, w + post)
        y = np.where((k >= 0) & (k < w), active, idle)
        self.c_trig.setData(np.append(k - 0.5, k[-1] + 0.5), y)
        self.c_trig_pts.setData(k, y)
        self.p_trig.setLabel("left", "Trigger", "V" if t.kind == "AO" else "logic")
        self.p_trig.setLabel("bottom", f"Samples from pulse start (1 sample = {dt_us:,.1f} µs)")
        self.p_trig.setTitle(f"Trigger pulse: {w} samples = {w * dt_us:,.1f} µs, "
                             f"{'falling' if low_active else 'rising'} edge")
        self.p_trig.setYRange(-0.1 * hi, 1.15 * hi)

    def _trigger_summary(self, a: AcquisitionConfig) -> str:
        t = self.get_trigger()
        kind = output_kind(t.physical)
        self.trg_kind.setText({"AO": "AO", "DO": "DO line", "PFI": "PFI"}.get(
            kind, "<span style='color:#c0392b'>unknown type</span>"))
        frame = t.frame_sync
        self.trg_level_v.setEnabled(kind == "AO" and not frame)
        for w in (self.trg_width, self.trg_timing, self.trg_pol):
            w.setEnabled(not frame or w is self.trg_pol)
        self.trg_queue.setEnabled(not frame and t.hw_timed)
        self.trg_sync.setEnabled(not frame and t.hw_timed)
        if a.fs > 0:
            self.trg_width_t.setText("(fixed by hardware)" if frame else f"= {t.pulse_samples / a.fs * 1e6:,.0f} µs")
        if not t.enabled:
            return ""
        if frame:
            return (f"<br><b>Frame sync:</b> AI sample clock → {t.physical}: one pulse per sample ({a.fs:g}/s) from sample 0. "
                    "Camera in external-sync mode ⇒ <b>frame f = sample f</b>, so every hit's trigger sample "
                    "(hit log) is also its frame number."
                    + ("" if kind == "PFI" else " <span style='color:#c0392b'><b>Choose a PFI terminal.</b></span>"))
        C = a.chunk_samples()
        if t.hw_timed:
            lo, hi = a.out_chunks() * C, (a.out_chunks() + 1) * C
            delay = (f"leaves {lo / a.fs * 1000:.0f}–{hi / a.fs * 1000:.0f} ms after the event is seen "
                     "(output queue), at a sample index that is logged exactly")
        else:
            delay = "leaves a few ms after the event is seen (software write; logged index approximate)"
        mi = t.mode_index()
        when = {0: "once at sample 0 (start of the acquisition)",
                1: f"for every detected hit; the hit is seen up to one read ({C / a.fs * 1000:.0f} ms) after it "
                   "crosses the level",
                2: f"for every accepted hit, i.e. once its record is complete ({(a.block_size - a.pre_samples) / a.fs:.2f} s "
                   "after the trigger) and it passed the checks — or when you click Accept",
                3: "for the first detected hit of every measurement point",
                4: "when a point has all its averages"}[mi]
        return (f"<br><b>External trigger:</b> pulse on {t.physical} {when}. It {delay}."
                + ("" if mi == 0 else " Set the camera to keep enough pre-trigger frames for this delay.")
                + " Pulse sample indices go to <i>&lt;base&gt;_trigger_log.csv</i>.")
