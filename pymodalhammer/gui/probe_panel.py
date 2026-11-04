# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Probe mode: tap the structure anywhere and see what every channel reads, without averaging or
saving. Level meters (live peak and peak hold, as % of each channel's range), the last hit in time,
the hammer's force spectrum (how far up the tip excites), and suggestions for the trigger level and
the input ranges."""

from __future__ import annotations

from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QProgressBar, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..config import ChannelConfig
from ..processing import force_rolloff
from .plots import COLOR_A, COLOR_B, DualAxisPlot

NONE_ITEM = "(none)"


def nice_ceil(x: float) -> float:
    """Round up to 1, 2 or 5 × 10^k."""
    if x <= 0:
        return 1.0
    e = np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * 10 ** e >= x * (1 - 1e-9):
            return float(m * 10 ** e)
    return float(10 ** (e + 1))


def _bar() -> QProgressBar:
    b = QProgressBar()
    b.setRange(0, 1000)
    b.setTextVisible(True)
    b.setAlignment(Qt.AlignCenter)
    return b


def _set_bar(b: QProgressBar, value: float, rng: float, unit: str):
    frac = value / rng if rng > 0 else 0
    b.setValue(int(min(1000, 1000 * frac)))
    b.setFormat(f"{value:.3g} {unit}  ({100 * frac:.0f} %)")
    col = "#27ae60" if frac < 0.7 else ("#f39c12" if frac < 0.95 else "#c0392b")
    b.setStyleSheet(f"QProgressBar {{ border: 1px solid #bbb; border-radius: 3px; text-align: center }}"
                    f"QProgressBar::chunk {{ background: {col}; }}")


class ProbePanel(QWidget):
    start_requested = Signal()
    stop_requested = Signal()
    simulate_hit = Signal()
    suggest_trigger = Signal(float)
    suggest_ranges = Signal(object)          # {channel name: range}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels: List[ChannelConfig] = []
        self.ref_i = -1
        self._hold = None
        self._hit_peaks: List[np.ndarray] = []
        self._last_hit = None
        self._scope = None
        self._level = 0.0
        self._level_u = ""

        intro = QLabel("Tap the structure where you plan to hit and where the responses are weakest. Nothing is "
                       "averaged or saved. Watch the meters: <span style='color:#27ae60'>green</span> &lt; 70 % of the "
                       "range, <span style='color:#f39c12'>orange</span> &lt; 95 %, "
                       "<span style='color:#c0392b'>red</span> = (almost) overloaded.")
        intro.setWordWrap(True)
        intro.setMinimumWidth(100)
        self.btn_start = QPushButton("▶  Start probe")
        self.btn_start.setMinimumHeight(34)
        self.btn_start.setStyleSheet("QPushButton { background:#8e44ad; color:white; font-weight:bold; "
                                     "border-radius:4px; padding:0 14px } QPushButton:disabled { background:#95a5a6 }")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setMinimumHeight(34)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton { background:#c0392b; color:white; font-weight:bold; "
                                    "border-radius:4px; padding:0 14px } QPushButton:disabled { background:#95a5a6 }")
        self.btn_reset = QPushButton("Reset peak hold")
        self.btn_sim = QPushButton("🔨 Simulate a hit")
        self.btn_sim.setToolTip("Simulation mode only: strike the simulated structure now")
        self.btn_sim.setVisible(False)
        self.btn_start.clicked.connect(self.start_requested)
        self.btn_stop.clicked.connect(self.stop_requested)
        self.btn_reset.clicked.connect(self.reset_hold)
        self.btn_sim.clicked.connect(self.simulate_hit)
        top = QHBoxLayout()
        for b in (self.btn_start, self.btn_stop, self.btn_reset, self.btn_sim):
            top.addWidget(b)
        top.addStretch()
        self.status = QLabel("Idle")
        self.status.setStyleSheet("font-weight: bold")
        top.addWidget(self.status)

        # ---------------- meters
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Channel", "Range ±", "Live peak (0.5 s)", "Peak hold"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)

        # ---------------- suggestions
        self.headroom = QDoubleSpinBox()
        self.headroom.setRange(1.1, 20)
        self.headroom.setDecimals(1)
        self.headroom.setPrefix("× ")
        self.headroom.setToolTip("Suggested range = this × the largest peak seen")
        self.trig_pct = QDoubleSpinBox()
        self.trig_pct.setRange(1, 90)
        self.trig_pct.setDecimals(0)
        self.trig_pct.setSuffix(" %")
        self.trig_pct.setToolTip("Suggested trigger level = this % of the median hammer peak")
        self.btn_trig = QPushButton("Use suggested trigger level")
        self.btn_trig.setEnabled(False)
        self.btn_rng = QPushButton("Apply suggested ranges")
        self.btn_rng.setEnabled(False)
        self.btn_rng.setToolTip("Writes the suggested ranges into the Channel Setup table")
        self.btn_trig.clicked.connect(self._emit_trig)
        self.btn_rng.clicked.connect(self._emit_ranges)
        self.suggest = QLabel()
        self.suggest.setWordWrap(True)
        self.suggest.setTextFormat(Qt.RichText)
        self.suggest.setStyleSheet("QLabel { background: palette(alternate-base); padding: 6px; }")
        sg = QGridLayout()
        sg.addWidget(QLabel("Trigger at"), 0, 0)
        sg.addWidget(self.trig_pct, 0, 1)
        sg.addWidget(QLabel("of median hit"), 0, 2)
        sg.addWidget(self.btn_trig, 0, 3)
        sg.addWidget(QLabel("Range headroom"), 1, 0)
        sg.addWidget(self.headroom, 1, 1)
        sg.addWidget(self.btn_rng, 1, 3)
        sg.setColumnStretch(4, 1)
        for w in (self.headroom, self.trig_pct):
            w.valueChanged.connect(self._update_suggestions)

        # ---------------- plots
        self.sel_a, self.sel_b = QComboBox(), QComboBox()
        self.sel_a.setStyleSheet(f"QComboBox {{ color: {COLOR_A} }}")
        self.sel_b.setStyleSheet(f"QComboBox {{ color: {COLOR_B} }}")
        for c in (self.sel_a, self.sel_b):
            c.currentIndexChanged.connect(self._redraw)
        self.p_scope = DualAxisPlot(downsample=True)
        self.p_scope.setLabel("bottom", "Time from start of acquisition", "s")
        self.p_scope.setTitle("Live (last 2 s)")
        self.lvl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#c0392b", style=Qt.DashLine))
        self.p_scope.addItem(self.lvl_line)
        self.p_hit = DualAxisPlot()
        self.p_hit.setLabel("bottom", "Time from trigger", "s")
        self.p_hit.setTitle("Last hit")
        self.p_fs = pg.PlotWidget(title="Hammer force spectrum (last hit)")
        self.p_fs.setLabel("bottom", "Frequency", "Hz")
        self.p_fs.setLabel("left", "Level", "dB re low-frequency level")
        self.p_fs.showGrid(x=True, y=True, alpha=0.3)
        self.p_fs.getAxis("bottom").enableAutoSIPrefix(False)
        self.c_fs = self.p_fs.plot(pen=pg.mkPen("#2b7bb9", width=1.5))
        for db in (-10, -20):
            self.p_fs.addItem(pg.InfiniteLine(pos=db, angle=0, pen=pg.mkPen("#7f8c8d", style=Qt.DashLine),
                                              label=f"{db} dB", labelOpts={"position": 0.95, "color": "#7f8c8d"}))
        self.f10 = pg.InfiniteLine(angle=90, pen=pg.mkPen("#27ae60", width=1.5), label="−10 dB",
                                   labelOpts={"position": 0.9, "color": "#27ae60"})
        self.p_fs.addItem(self.f10)
        self.p_fs.setYRange(-40, 5)
        self.hit_info = QLabel()
        self.hit_info.setWordWrap(True)
        self.pulse_info = QLabel()
        self.pulse_info.setWordWrap(True)
        self.pulse_info.setTextFormat(Qt.RichText)
        self._marks = []                     # pulse markers drawn in the plots

        sel = QHBoxLayout()
        sel.addWidget(QLabel("<b>Channels</b>  A:"))
        sel.addWidget(self.sel_a, 1)
        sel.addWidget(QLabel("B:"))
        sel.addWidget(self.sel_b, 1)
        plots = QSplitter(Qt.Vertical)
        plots.addWidget(self.p_scope)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.p_hit)
        bottom.addWidget(self.p_fs)
        plots.addWidget(bottom)
        plots.setSizes([200, 260])
        meters = QWidget()
        ml = QVBoxLayout(meters)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.table, 1)
        ml.addLayout(sg)
        ml.addWidget(self.suggest)
        ml.addWidget(self.hit_info)
        ml.addWidget(self.pulse_info)
        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addLayout(sel)
        pl.addWidget(plots, 1)
        main = QSplitter(Qt.Vertical)
        main.addWidget(meters)
        main.addWidget(pw)
        main.setSizes([360, 440])
        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addLayout(top)
        lay.addWidget(main, 1)

    # ------------------------------------------------------------------ setup
    def set_suggest_params(self, headroom: float, trig_pct: float):
        self.headroom.setValue(headroom)
        self.trig_pct.setValue(trig_pct)

    def suggest_params(self):
        return self.headroom.value(), self.trig_pct.value()

    def set_trigger_info(self, level: float, units: str):
        self._level, self._level_u = level, units

    def prepare(self, channels: List[ChannelConfig], ref_i: int, trig_i: int, simulate: bool, trig=None):
        self.channels = channels
        on = trig is not None and trig.enabled and trig.in_probe
        self.trig = trig if on else None
        self._trig_wave = on and trig.hw_timed and not trig.frame_sync      # a waveform to show
        self._pulses = []
        self.ref_i = ref_i
        self.trig_i = trig_i
        self.btn_sim.setVisible(simulate)
        self._hold = np.zeros(len(channels))
        self._hit_peaks = []
        self._last_hit = None
        self._scope = None
        self.table.setRowCount(len(channels))
        self.live_bars, self.hold_bars = [], []
        for i, c in enumerate(channels):
            it = QTableWidgetItem(f"{c.name}" + ("  (hammer)" if i == ref_i else ""))
            self.table.setItem(i, 0, it)
            self.table.setItem(i, 1, QTableWidgetItem(f"{c.max_range:g} {c.units}"))
            lb, hb = _bar(), _bar()
            self.table.setCellWidget(i, 2, lb)
            self.table.setCellWidget(i, 3, hb)
            self.live_bars.append(lb)
            self.hold_bars.append(hb)
            _set_bar(lb, 0, c.max_range, c.units)
            _set_bar(hb, 0, c.max_range, c.units)
        names = [f"{c.name}  [{c.units}]" for c in channels]
        if self._trig_wave:
            names.append("Trigger output  [" + ("V" if trig.kind == "AO" else "logic") + "]")
        b_default = len(channels) + 1 if self._trig_wave else (2 if len(channels) > 1 else 0)
        for combo, extra, default in ((self.sel_a, [], ref_i if ref_i >= 0 else 0),
                                      (self.sel_b, [NONE_ITEM], b_default)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + names)
            combo.setCurrentIndex(min(default, combo.count() - 1))
            combo.blockSignals(False)
        self.btn_trig.setEnabled(False)
        self.btn_rng.setEnabled(False)
        self.hit_info.setText("")
        if trig is None or not trig.enabled:
            self.pulse_info.setText("External trigger: off.")
        elif not trig.in_probe:
            self.pulse_info.setText("External trigger: not sent in probe mode — tick <i>Also send pulses in Probe "
                                    "mode</i> in the External trigger box to see the pulses here.")
        elif trig.frame_sync:
            self.pulse_info.setText(f"External trigger: frame sync on {trig.physical} (one pulse per sample) — "
                                    "nothing to mark.")
        else:
            self.pulse_info.setText(f"External trigger on {trig.physical}: <i>{trig.mode}</i>. "
                                    + ("The output is channel <b>Trigger output</b> (B); " if self._trig_wave else
                                       "On-demand: no waveform is recorded; ")
                                    + "<span style='color:#8e44ad'>purple lines</span> mark the pulses.")
        self.suggest.setText("Tap the structure a few times: suggestions appear here.")
        self.c_fs.setData([], [])
        self.p_hit.set_curves([], [])

    def set_running(self, on: bool):
        self.btn_start.setEnabled(not on)
        self.btn_stop.setEnabled(on)
        self.btn_sim.setEnabled(on)
        self.status.setText("Probing — tap away" if on else "Idle")

    def reset_hold(self):
        if self._hold is not None:
            self._hold[:] = 0
        self._hit_peaks = []
        self._update_suggestions()

    # ------------------------------------------------------------------ data
    def _sel(self, combo):
        """Channel index, len(channels) for the trigger output, or None."""
        i = combo.currentIndex() - (1 if combo is self.sel_b else 0)
        n = len(self.channels) + (1 if getattr(self, "_trig_wave", False) else 0)
        if i < 0 or i >= n:
            return None
        return i

    def _label(self, i):
        if i < len(self.channels):
            return self.channels[i].name, self.channels[i].units
        return "Trigger output", ("V" if self.trig.kind == "AO" else "")

    @Slot(object)
    def on_pulse(self, rec: dict):
        """A pulse left the trigger line (or was switched on demand)."""
        self._pulses = (self._pulses + [rec])[-50:]
        self._update_pulse_info()
        self._redraw()

    def _update_pulse_info(self):
        if not self._pulses:
            return
        rec = self._pulses[-1]
        h = self._last_hit
        rel = ""
        if h is not None and rec.get("hit") and rec["hit"] == h.number:
            rel = f", <b>{(rec['sample'] - h.sample) / h.fs * 1000:.0f} ms after the hit's trigger point</b>"
        self.pulse_info.setText(
            f"<span style='color:#8e44ad'><b>Pulse {rec['pulse']}</b></span> ({rec['reason']}) on "
            f"{self.trig.physical if self.trig else ''} at sample {rec['sample']}"
            + ("" if rec["exact"] else " (approx.)") + rel
            + f"; {rec['latency_ms']:.0f} ms after the software saw the event.")

    @Slot(object, int, float, object, int, object)
    def on_scope(self, data: np.ndarray, start: int, fs: float, peaks, backlog: int, trig=None):
        if not self.channels or data.shape[0] != len(self.channels):
            return
        self._scope = (data, start, fs, trig)
        m = int(min(data.shape[1], 0.5 * fs))
        live = np.max(np.abs(data[:, -m:]), axis=1)
        self._hold = np.maximum(self._hold, live)
        for i, c in enumerate(self.channels):
            _set_bar(self.live_bars[i], live[i], c.max_range, c.units)
            _set_bar(self.hold_bars[i], self._hold[i], c.max_range, c.units)
        self._redraw()

    @Slot(object)
    def on_hit(self, hit):
        self._last_hit = hit
        self._hit_peaks.append(hit.peaks)
        self._hold = np.maximum(self._hold, hit.peaks)
        msg = [f"<b>Hit {len(self._hit_peaks)}</b> at t = {hit.time_s:.2f} s"]
        if self.ref_i >= 0:
            c = self.channels[self.ref_i]
            X = np.fft.rfft(hit.data[self.ref_i]) / hit.fs
            fr = np.arange(X.size) * hit.fs / hit.data.shape[1]
            a = np.abs(X)
            k = max(1, int(0.02 * a.size))
            lev = 20 * np.log10(a / (np.max(a[1:k + 1]) + 1e-30) + 1e-30)
            self.c_fs.setData(fr[1:], lev[1:])
            f10, f20 = force_rolloff(fr, X, -10), force_rolloff(fr, X, -20)
            self.f10.setPos(f10)
            self.p_fs.setXRange(0, min(fr[-1], max(3 * f20, 4 * f10)), padding=0)
            msg.append(f"hammer peak <b>{hit.peaks[self.ref_i]:.4g} {c.units}</b>; tip excites up to "
                       f"<b>{f10:.0f} Hz</b> (−10 dB), {f20:.0f} Hz (−20 dB)")
        bad = [ch.message for ch in hit.checks if not ch.ok]
        msg.append(("<span style='color:#c0392b'>" + "; ".join(bad) + "</span>") if bad else
                   "<span style='color:#1e8449'>checks ok</span>")
        self.hit_info.setText(" · ".join(msg))
        self._update_pulse_info()
        self._update_suggestions()
        self._redraw()

    def _redraw(self, *_):
        a, b = self._sel(self.sel_a), self._sel(self.sel_b)
        if a is None:
            return
        la, lb = self._label(a), (self._label(b) if b is not None else ("", ""))
        for p, m in self._marks:
            p.removeItem(m)
        self._marks = []
        n = len(self.channels)

        def mark(plot, x, text=""):
            ln = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen("#8e44ad", width=1.5, style=Qt.DashLine),
                                 label=text, labelOpts={"position": 0.92, "color": "#8e44ad"})
            plot.addItem(ln)
            self._marks.append((plot, ln))

        if self._scope is not None:
            data, start, fs, trig = self._scope

            def sig(i):
                if i < n:
                    return data[i]
                return trig if trig is not None else np.zeros(data.shape[1])

            t = (start + np.arange(data.shape[1])) / fs
            self.p_scope.set_curves(t, sig(a), *la, sig(b) if b is not None else None, *lb)
            show = a == self.trig_i and self._level > 0
            self.lvl_line.setVisible(show)
            if show:
                self.lvl_line.setPos(self._level)
            for rec in self._pulses:
                if start <= rec["sample"] < start + data.shape[1]:
                    mark(self.p_scope, rec["sample"] / fs, f"pulse {rec['pulse']}")
        h = self._last_hit
        if h is not None:
            def hsig(i):
                if i < n:
                    return h.data[i]
                return h.trig_wave if h.trig_wave is not None else np.zeros(h.data.shape[1])

            t = (np.arange(h.data.shape[1]) - h.pre) / h.fs
            self.p_hit.set_curves(t, hsig(a), *la, hsig(b) if b is not None else None, *lb)
            t_end = (h.data.shape[1] - h.pre) / h.fs
            for rec in self._pulses:
                if rec.get("hit") == h.number:
                    x = (rec["sample"] - h.sample) / h.fs
                    mark(self.p_hit, x, f"pulse {rec['pulse']}" + ("" if x < t_end else " →"))
            pts = [(r["sample"] - h.sample) / h.fs for r in self._pulses if r.get("hit") == h.number]
            if pts and max(pts) > t_end:          # pulse after the record (long output queue): widen the view
                self.p_hit.setXRange(t[0], max(pts) * 1.05, padding=0)
            else:
                self.p_hit.enableAutoRange(axis="x")

    def _update_suggestions(self, *_):
        if not self.channels:
            return
        head, pct = self.headroom.value(), self.trig_pct.value()
        parts = []
        if self._hit_peaks and self.ref_i >= 0:
            med = float(np.median([p[self.ref_i] for p in self._hit_peaks]))
            c = self.channels[self.ref_i]
            self._sugg_trig = float(f"{med * pct / 100:.3g}")
            parts.append(f"<b>Trigger level</b>: {self._sugg_trig:g} {c.units} ({pct:.0f} % of the median of "
                         f"{len(self._hit_peaks)} hit(s), {med:.4g} {c.units}); now {self._level:g} {self._level_u}.")
            self.btn_trig.setEnabled(True)
        peak = self._hold if self._hold is not None else np.zeros(len(self.channels))
        self._sugg_rng = {}
        rows = []
        for c, p in zip(self.channels, peak):
            if p <= 0:
                continue
            r = nice_ceil(head * p)
            self._sugg_rng[c.name] = r
            flag = (" <span style='color:#c0392b'>↑</span>" if r > c.max_range else
                    (" ↓" if r < c.max_range / 4 else ""))
            rows.append(f"{c.name} ±{r:g} {c.units}{flag}")
        if rows:
            parts.append("<b>Ranges</b> (" + f"{head:g} × peak hold): " + ", ".join(rows)
                         + ". ↑ = more than the current range (overload risk), ↓ = less than ¼ of it (resolution). "
                           "The module's own input range (e.g. ±5 V on the NI 9234) is fixed; the range here sets "
                           "the scaling and the overload check.")
            self.btn_rng.setEnabled(True)
        self.suggest.setText("<br>".join(parts) if parts else "Tap the structure a few times: suggestions appear here.")

    def _emit_trig(self):
        if getattr(self, "_sugg_trig", 0) > 0:
            self.suggest_trigger.emit(self._sugg_trig)

    def _emit_ranges(self):
        if getattr(self, "_sugg_rng", None):
            self.suggest_ranges.emit(dict(self._sugg_rng))
