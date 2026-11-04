# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Compare runs: overlay the FRF, amplitude spectrum or time data of the same channel from several saved
measurement points / runs. Channels are matched by name; every run uses its own hammer reference,
so in a roving-hammer test each curve is one hammer point."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..results import amplitude_spectra, find_runs, frf_table, scan_run, time_records
from .plots import export_dialog

PALETTE = ["#2b7bb9", "#d35400", "#27ae60", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#f39c12",
           "#2c3e50", "#e84393"]


class CompareTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs: Dict[str, object] = {}

        self.list = QListWidget()
        self.list.setToolTip("Tick the runs to overlay")
        self.list.itemChanged.connect(self._refresh_items)
        b_add = QPushButton("＋ Add runs from folder…")
        b_add.clicked.connect(self._add_folder)
        b_all, b_none = QPushButton("All"), QPushButton("None")
        b_all.clicked.connect(lambda: self._check_all(True))
        b_none.clicked.connect(lambda: self._check_all(False))
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Runs</b> (tick to overlay)"))
        ll.addWidget(self.list, 1)
        bl = QHBoxLayout()
        bl.addWidget(b_all)
        bl.addWidget(b_none)
        ll.addLayout(bl)
        ll.addWidget(b_add)

        self.qty = QComboBox()
        self.qty.addItems(["FRF", "Amplitude spectrum", "Time data"])
        # time data options
        self.hitsel = QComboBox()
        self.hitsel.addItems(["First averaged hit", "Every averaged hit", "Hit number", "Mean of averaged hits"])
        self.hitsel.setToolTip("Which hit(s) of every run to draw. All records are aligned at the trigger point (t = 0).")
        self.hitnum = QSpinBox()
        self.hitnum.setRange(1, 100000)
        self.hitnum.setPrefix("# ")
        self.norm = QCheckBox("÷ hammer peak")
        self.norm.setToolTip("Divide every record by its hit's peak hammer force, so hits of different strength compare")
        self.twin = QCheckBox("windowed")
        self.twin.setToolTip("Apply each run's force / exponential window, as used for its FRF")
        self.tzoom = QCheckBox("zoom on impact")
        self.tzoom.setToolTip("Show the first 20 ms after the trigger")
        for w in (self.norm, self.twin, self.tzoom):
            w.toggled.connect(self._draw)
        self.hitsel.currentIndexChanged.connect(self._draw)
        self.hitnum.valueChanged.connect(self._draw)
        self.item = QComboBox()
        self.item.setToolTip("Channel name; matched across runs")
        self.est = QComboBox()
        self.est.addItems(["H1", "H2", "Hv"])
        self.tdms = QCheckBox("Recompute from TDMS when available")
        self.tdms.setChecked(True)
        self.tdms.setToolTip("FRFs from the runs' saved hits (the ones that were averaged); otherwise the stored UFF files")
        self.db = QCheckBox("dB")
        self.db.setToolTip("Magnitude in dB re 1 instead of a log axis")
        self.info = QLabel()
        self.info.setWordWrap(True)
        exp = QPushButton("📷 Export image…")
        self.qty.currentIndexChanged.connect(self._refresh_items)
        self.tdms.toggled.connect(self._refresh_items)
        for w in (self.item, self.est):
            w.currentIndexChanged.connect(self._draw)
        self.db.toggled.connect(self._draw)

        self.p_q, self.p_mag, self.p_ph = pg.PlotWidget(), pg.PlotWidget(), pg.PlotWidget()
        self.p_ph.setLabel("left", "Phase", "°")
        self.p_ph.setYRange(-180, 180)
        self.p_ph.setLabel("bottom", "Frequency", "Hz")
        for p in (self.p_q, self.p_mag, self.p_ph):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            if p is not self.p_mag:               # log |H| axis keeps pyqtgraph's own tick labelling
                p.getAxis("left").enableAutoSIPrefix(False)
        self.p_ph.setXLink(self.p_mag)
        self.p_q.setXLink(self.p_mag)
        self.legend = self.p_mag.addLegend(offset=(-10, 10))
        self.plots = QSplitter(Qt.Vertical)
        for p in (self.p_q, self.p_mag, self.p_ph):
            self.plots.addWidget(p)
        self.plots.setSizes([140, 320, 160])
        exp.clicked.connect(lambda: export_dialog(self, self.plots, "compare_runs.png"))

        ctl = QHBoxLayout()
        for lbl, w in (("Show", self.qty), ("  Channel", self.item), ("  Estimator", self.est)):
            ql = QLabel(lbl)
            ctl.addWidget(ql)
            ctl.addWidget(w, 1 if w is self.item else 0)
            if w is self.est:
                self.est_label = ql
        ctl.addWidget(exp)
        ctl2 = QHBoxLayout()                       # row 2: options of the chosen view (keeps the tab narrow)
        ctl2.addWidget(self.tdms)
        self.time_box = QWidget()
        tb = QHBoxLayout(self.time_box)
        tb.setContentsMargins(0, 0, 0, 0)
        for w in (self.hitsel, self.hitnum, self.norm, self.twin, self.tzoom):
            tb.addWidget(w)
        self.time_opts = [self.time_box]
        ctl2.addWidget(self.time_box)
        ctl2.addWidget(self.db)
        ctl2.addStretch()
        self._ctl2 = ctl2
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(ctl)
        rl.addLayout(ctl2)
        rl.addWidget(self.info)
        rl.addWidget(self.plots, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(lw)
        split.addWidget(rw)
        split.setSizes([300, 1000])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self._refresh_items()                      # show only the options of the current view

    # ------------------------------------------------------------------ runs
    def set_runs(self, folders: List[str]):
        have = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())}
        self.list.blockSignals(True)
        for f in folders:
            if f not in have:
                self._add_item(f)
        self.list.blockSignals(False)

    def _add_item(self, folder: str, checked: bool = False):
        run = scan_run(folder)
        self.runs[folder] = run
        it = QListWidgetItem(f"[{run.point_label}] {run.name}")
        it.setData(Qt.UserRole, folder)
        it.setToolTip(folder)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.list.addItem(it)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with runs")
        if d:
            self.set_runs(find_runs(d))

    def _check_all(self, on: bool):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(False)
        self._refresh_items()

    def check(self, folders: List[str]):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in folders else Qt.Unchecked)
        self.list.blockSignals(False)
        self._refresh_items()

    def _selected(self):
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                f = it.data(Qt.UserRole)
                run = self.runs.get(f) or scan_run(f)
                self.runs[f] = run
                if run.ok:
                    out.append((it.text(), run))
        return out

    # ------------------------------------------------------------------ data
    def _table(self, run) -> Dict[str, Dict]:
        key = (run.folder, self.qty.currentIndex(), self.tdms.isChecked())
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            try:
                q = self.qty.currentIndex()
                cache[key] = (amplitude_spectra(run) if q == 1 else time_records(run) if q == 2
                              else frf_table(run, self.tdms.isChecked()))
            except Exception as e:
                cache[key] = {}
                self.info.setText(f"{run.name}: {e}")
        return cache[key]

    def _refresh_items(self, *_):
        spec = self.qty.currentIndex() == 1
        timed = self.qty.currentIndex() == 2
        for w in (self.est, self.est_label, self.tdms):
            w.setVisible(not spec and not timed)
        for w in self.time_opts:
            w.setVisible(timed)
        self.db.setVisible(not timed)
        self._ctl2.invalidate()                    # hidden options must not keep their width reserved
        names: List[str] = []
        for _, run in self._selected():
            for n in self._table(run):
                if n not in names:
                    names.append(n)
        cur = self.item.currentText()
        self.item.blockSignals(True)
        self.item.clear()
        self.item.addItems(names)
        if cur in names:
            self.item.setCurrentText(cur)
        self.item.blockSignals(False)
        self._draw()

    def _draw(self, *_):
        for p in (self.p_q, self.p_mag, self.p_ph):
            for it in list(p.listDataItems()):
                p.removeItem(it)
        self.legend.clear()
        if self.qty.currentIndex() == 2:
            return self._draw_time()
        self.p_mag.setLabel("bottom", "")
        self.p_mag.getAxis("left").enableAutoSIPrefix(True)
        self.p_mag.enableAutoRange()
        spec = self.qty.currentIndex() == 1
        db = self.db.isChecked()
        self.p_mag.setLogMode(y=not db)
        self.p_q.setVisible(not spec)
        self.p_ph.setVisible(not spec)
        self.p_mag.setLabel("bottom", "Frequency" if spec else "", "Hz" if spec else "")
        name, est = self.item.currentText(), self.est.currentText()
        sel = self._selected()
        if not name or not sel:
            self.info.setText("Tick two or more runs on the left and choose a channel.")
            return
        shown, skipped, units = 0, [], set()
        for i, (label, run) in enumerate(sel):
            d = self._table(run).get(name)
            if d is None:
                skipped.append(f"{label} (no '{name}')")
                continue
            color = PALETTE[i % len(PALETTE)]
            pen = pg.mkPen(color, width=1.5)
            y = d["amp"] if spec else d[est]
            if y is None:
                skipped.append(f"{label} (no {est})")
                continue
            f = d["freqs"]
            ok = np.isfinite(y) & (f > 0)
            mag = np.abs(y[ok]) + 1e-30
            self.p_mag.plot(f[ok], 20 * np.log10(mag) if db else mag, pen=pen, name=label)
            if not spec:
                self.p_ph.plot(f[ok], np.degrees(np.angle(y[ok])), pen=pen)
                if d.get("q") is not None:
                    self.p_q.plot(f[ok], np.real(d["q"][ok]), pen=pen)
            units.add(d["units"])
            shown += 1
        u = " / ".join(sorted(x for x in units if x))
        lab = f"|{name}|" if spec else f"|{est}|"
        self.p_mag.setLabel("left", lab + (" (dB)" if db else (f"  [{u}]" if u else "")))
        self.p_q.setLabel("left", "Coherence γ²")
        self.p_q.setYRange(0, 1.05)
        msg = f"{shown} run(s) overlaid."
        if skipped:
            msg += " Not shown: " + "; ".join(skipped)
        if len(units) > 1:
            msg += "  ⚠ different units across runs"
        self.info.setText(msg)

    def _draw_time(self):
        self.p_mag.setLogMode(y=False)
        self.p_mag.getAxis("left").enableAutoSIPrefix(False)
        self.p_q.setVisible(False)
        self.p_ph.setVisible(False)
        self.p_mag.setLabel("bottom", "Time from trigger", "s")
        name, mode, num = self.item.currentText(), self.hitsel.currentIndex(), self.hitnum.value()
        self.hitnum.setEnabled(mode == 2)
        sel = self._selected()
        if not name or not sel:
            self.info.setText("Tick two or more runs on the left and choose a channel.")
            return
        shown, skipped, units = 0, [], set()
        t0, t_end = 0.0, 0.0
        for i, (label, run) in enumerate(sel):
            d = self._table(run).get(name)
            if d is None:
                skipped.append(f"{label} ({'no saved hits' if not self._table(run) else 'no ' + repr(name)})")
                continue
            used = [h for h in d["hits"] if h["included"]]
            if mode == 0:
                chosen = used[:1]
            elif mode == 1 or mode == 3:
                chosen = used
            else:
                chosen = [h for h in d["hits"] if h["number"] == num]
            if not chosen:
                skipped.append(f"{label} (no {'hit #' + str(num) if mode == 2 else 'averaged hit'})")
                continue
            ys = []
            for h in chosen:
                y = h["y"] * (d["win"] if self.twin.isChecked() else 1.0)
                if self.norm.isChecked():
                    y = y / (h["ref_peak"] or 1.0)
                ys.append(y)
            color = PALETTE[i % len(PALETTE)]
            if mode == 3:
                ys = [np.mean(ys, axis=0)]
                label += f"  (mean of {len(chosen)})"
            elif mode == 2:
                label += f"  (hit {num})"
            for k, y in enumerate(ys):
                c = pg.mkColor(color)
                if len(ys) > 1:
                    c.setAlpha(140)
                it = self.p_mag.plot(d["t"], y, pen=pg.mkPen(c, width=1.2), name=label if k == 0 else None)
                it.setDownsampling(auto=True, method="peak")
                it.setClipToView(True)
            t0, t_end = min(t0, float(d["t"][0])), max(t_end, float(d["t"][-1]))
            units.add(d["units"] + (f"/{d['ref_units']}" if self.norm.isChecked() else ""))
            shown += 1
        u = " / ".join(sorted(x for x in units if x))
        self.p_mag.setLabel("left", f"{name}" + (f"  [{u}]" if u else ""))
        self.p_mag.enableAutoRange(axis="y")
        if self.tzoom.isChecked():
            self.p_mag.setXRange(-0.002, 0.02, padding=0)
        elif shown:
            self.p_mag.setXRange(t0, t_end, padding=0.01)
        msg = f"{shown} run(s) overlaid (t = 0 at each hit's trigger point)."
        if skipped:
            msg += " Not shown: " + "; ".join(skipped)
        if len(units) > 1:
            msg += "  ⚠ different units across runs"
        self.info.setText(msg)
