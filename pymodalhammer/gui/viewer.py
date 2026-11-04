# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Data Viewer tab: browse saved measurement points. FRFs (recomputed from the ticked hits or as
stored in the UFF files), time data of every hit, amplitude spectra, re-processing with other
windows / another choice of hits, and comparison of several runs."""

from __future__ import annotations

import copy
import os
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from ..config import FORCE_WINDOWS, RESP_WINDOWS
from ..processing import STATUS_COLORS, Windows, hit_spectra
from ..results import Run, find_runs, frf_table, load_hits, recompute, scan_run
from .compare import CompareTab
from .plots import COLOR_A, COLOR_B, EST_COLORS, FRFStack, export_dialog

SRC_TICKED = "Recomputed from the ticked hits (TDMS)"
SRC_STORED = "Stored UFF files"


class ViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run: Optional[Run] = None
        self._res = None

        # ---------------- folder / run
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("A run folder, or a folder containing run folders")
        self.folder.returnPressed.connect(self.rescan)
        browse = QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse)
        rescan = QPushButton("↻ Scan")
        rescan.clicked.connect(self.rescan)
        self.run_sel = QComboBox()
        self.run_sel.currentIndexChanged.connect(self._load_run)
        self.summary = QLabel("Choose a folder to view saved results.")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setStyleSheet("QLabel { background: palette(alternate-base); padding: 6px; }")
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Folder</b>"))
        top.addWidget(self.folder, 1)
        top.addWidget(browse)
        top.addWidget(rescan)
        top.addWidget(QLabel("Run"))
        top.addWidget(self.run_sel, 1)

        # ---------------- hits + re-processing (left)
        self.hits = QListWidget()
        self.hits.setToolTip("Ticked hits are used for 'Recomputed' FRFs and the spectra; the selected one is drawn bold")
        self.hits.itemChanged.connect(self._hits_changed)
        self.hits.currentRowChanged.connect(lambda _: self._draw())
        b_avg, b_all, b_none = QPushButton("Averaged"), QPushButton("All"), QPushButton("None")
        b_avg.setToolTip("Tick the hits that were averaged during the measurement")
        b_avg.clicked.connect(lambda: self._tick(lambda h: h["included"]))
        b_all.clicked.connect(lambda: self._tick(lambda h: True))
        b_none.clicked.connect(lambda: self._tick(lambda h: False))
        self.rp_fwin = QComboBox()
        self.rp_fwin.addItems(FORCE_WINDOWS)
        self.rp_fw = QDoubleSpinBox()
        self.rp_fw.setRange(0.1, 10000)
        self.rp_fw.setSuffix(" ms")
        self.rp_ft = QDoubleSpinBox()
        self.rp_ft.setRange(0, 1000)
        self.rp_ft.setSuffix(" ms")
        self.rp_rwin = QComboBox()
        self.rp_rwin.addItems(RESP_WINDOWS)
        self.rp_end = QDoubleSpinBox()
        self.rp_end.setRange(0.01, 100)
        self.rp_end.setSuffix(" %")
        self.rp_info = QLabel()
        self.rp_info.setWordWrap(True)
        self.rp_info.setStyleSheet("color: gray")
        b_apply = QPushButton("Apply windows")
        b_reset = QPushButton("As measured")
        b_apply.clicked.connect(self._apply_windows)
        b_reset.clicked.connect(self._reset_windows)
        b_save = QPushButton("💾 Save FRFs of ticked hits as UFF…")
        b_save.setToolTip("Writes H1/H2/Hv/coherence of the ticked hits with these windows into a folder you choose; "
                          "the run itself is never changed")
        b_save.clicked.connect(self._save_reprocessed)
        g_rp = QGroupBox("Re-process")
        f = QFormLayout(g_rp)
        f.addRow("Hammer", self.rp_fwin)
        hl = QHBoxLayout()
        hl.addWidget(self.rp_fw)
        hl.addWidget(QLabel("+ taper"))
        hl.addWidget(self.rp_ft)
        f.addRow("  width", hl)
        f.addRow("Responses", self.rp_rwin)
        f.addRow("  end value", self.rp_end)
        bh = QHBoxLayout()
        bh.addWidget(b_apply)
        bh.addWidget(b_reset)
        f.addRow(bh)
        f.addRow(self.rp_info)
        f.addRow(b_save)
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Hits</b> (tick = use)"))
        ll.addWidget(self.hits, 1)
        bl = QHBoxLayout()
        for b in (b_avg, b_all, b_none):
            bl.addWidget(b)
        ll.addLayout(bl)
        ll.addWidget(g_rp)

        # ---------------- FRF page
        self.src = QComboBox()
        self.src.addItems([SRC_TICKED, SRC_STORED])
        self.src.currentIndexChanged.connect(self._fill_resp)
        self.resp = QComboBox()
        self.resp.currentIndexChanged.connect(self._draw)
        self.est_cb = {k: QCheckBox(k) for k in ("H1", "H2", "Hv")}
        for k, cb in self.est_cb.items():
            cb.setChecked(k == "H1")
            cb.setStyleSheet(f"QCheckBox {{ color: {EST_COLORS[k]}; font-weight: bold }}")
            cb.toggled.connect(self._draw)
        self.frf_info = QLabel()
        self.frf = FRFStack()
        exp1 = QPushButton("📷 Export image…")
        exp1.clicked.connect(lambda: export_dialog(self, self.frf, self._img("FRF")))
        fh = QHBoxLayout()
        fh.addWidget(QLabel("Source"))
        fh.addWidget(self.src)
        fh.addWidget(QLabel("  Response"))
        fh.addWidget(self.resp, 1)
        for cb in self.est_cb.values():
            fh.addWidget(cb)
        fh.addWidget(self.frf_info)
        fh.addWidget(exp1)
        frf_page = QWidget()
        fl = QVBoxLayout(frf_page)
        fl.addLayout(fh)
        fl.addWidget(self.frf, 1)

        # ---------------- time page
        self.ch_a, self.ch_b = QComboBox(), QComboBox()
        self.ch_a.setStyleSheet(f"QComboBox {{ color: {COLOR_A} }}")
        self.ch_b.setStyleSheet(f"QComboBox {{ color: {COLOR_B} }}")
        for c in (self.ch_a, self.ch_b):
            c.currentIndexChanged.connect(self._draw)
        self.t_win = QCheckBox("show windows")
        self.t_win.setChecked(True)
        self.t_win.toggled.connect(self._draw)
        self.t_trig = QCheckBox("trigger output")
        self.t_trig.setToolTip("Overlay the external trigger output (hardware-timed runs) on channel A, scaled")
        self.t_trig.toggled.connect(self._draw)
        self.p_ta, self.p_tb = pg.PlotWidget(), pg.PlotWidget()
        for p in (self.p_ta, self.p_tb):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.getAxis("left").enableAutoSIPrefix(False)
        self.p_tb.setXLink(self.p_ta)
        self.p_tb.setLabel("bottom", "Time from trigger", "s")
        exp2 = QPushButton("📷 Export image…")
        tsp = QSplitter(Qt.Vertical)
        tsp.addWidget(self.p_ta)
        tsp.addWidget(self.p_tb)
        exp2.clicked.connect(lambda: export_dialog(self, tsp, self._img("time")))
        th = QHBoxLayout()
        th.addWidget(QLabel("A"))
        th.addWidget(self.ch_a, 1)
        th.addWidget(QLabel("B"))
        th.addWidget(self.ch_b, 1)
        th.addWidget(self.t_win)
        th.addWidget(self.t_trig)
        th.addWidget(exp2)
        time_page = QWidget()
        tl = QVBoxLayout(time_page)
        tl.addLayout(th)
        tl.addWidget(tsp, 1)

        # ---------------- spectra page
        self.sp_ch = QComboBox()
        self.sp_ch.currentIndexChanged.connect(self._draw)
        self.sp_win = QCheckBox("windowed")
        self.sp_win.setChecked(True)
        self.sp_db = QCheckBox("dB")
        for w in (self.sp_win, self.sp_db):
            w.toggled.connect(self._draw)
        self.p_sp = pg.PlotWidget()
        self.p_sp.showGrid(x=True, y=True, alpha=0.3)
        self.p_sp.setLabel("bottom", "Frequency", "Hz")
        self.p_sp.getAxis("bottom").enableAutoSIPrefix(False)
        self.p_sp.getAxis("left").enableAutoSIPrefix(False)
        exp3 = QPushButton("📷 Export image…")
        exp3.clicked.connect(lambda: export_dialog(self, self.p_sp, self._img("spectrum")))
        sh = QHBoxLayout()
        sh.addWidget(QLabel("Channel"))
        sh.addWidget(self.sp_ch, 1)
        sh.addWidget(self.sp_win)
        sh.addWidget(self.sp_db)
        sh.addWidget(exp3)
        sp_page = QWidget()
        sl = QVBoxLayout(sp_page)
        sl.addLayout(sh)
        sp_lbl = QLabel("Fourier transform |X(f)| of every ticked hit (faint), the selected hit (bold) and their "
                        "RMS average (black).")
        sp_lbl.setWordWrap(True)
        sp_lbl.setMinimumWidth(100)
        sl.addWidget(sp_lbl)
        sl.addWidget(self.p_sp, 1)

        self.compare = CompareTab()
        self.tabs = QTabWidget()
        self.tabs.addTab(frf_page, "FRF")
        self.tabs.addTab(time_page, "Time data")
        self.tabs.addTab(sp_page, "Spectra")
        self.tabs.addTab(self.compare, "Compare runs")
        self.tabs.currentChanged.connect(self._tab_changed)

        self.hit_panel = lw
        split = QSplitter(Qt.Horizontal)
        split.addWidget(lw)
        split.addWidget(self.tabs)
        split.setSizes([300, 1100])
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.summary)
        lay.addWidget(split, 1)
        self._acq = None

    def _tab_changed(self, i: int):
        self.hit_panel.setVisible(self.tabs.widget(i) is not self.compare)   # Compare has its own run list
        self._draw()

    # ------------------------------------------------------------------ folders
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Run folder (or folder of runs)", self.folder.text() or os.getcwd())
        if d:
            self.folder.setText(d)
            self.rescan()

    def open_folder(self, folder: str, parent: str = ""):
        """Show `folder` (a run); offer its siblings in the parent folder too."""
        self.folder.setText(parent or folder)
        self.rescan(select=folder)

    def rescan(self, select: str = ""):
        runs = find_runs(self.folder.text().strip())
        self.run_sel.blockSignals(True)
        self.run_sel.clear()
        for r in runs:
            self.run_sel.addItem(os.path.basename(os.path.normpath(r)), r)
        self.run_sel.blockSignals(False)
        self.compare.set_runs(runs)
        if not runs:
            self.summary.setText("No runs found here (a run folder holds a setup.json).")
            return
        i = self.run_sel.findData(select) if select else self.run_sel.count() - 1
        self.run_sel.setCurrentIndex(max(i, 0))
        self._load_run()

    def _load_run(self, *_):
        f = self.run_sel.currentData()
        if not f:
            return
        self.run = run = scan_run(f)
        self.summary.setText(run.summary_html())
        if not run.ok:
            return
        try:
            hits = load_hits(run)
        except Exception as e:
            hits = []
            self.summary.setText(run.summary_html() + f"<br><span style='color:#c0392b'>Cannot read hits: {e}</span>")
        self.hits.blockSignals(True)
        self.hits.clear()
        ref = run.ref_i
        for h in hits:
            pk = h["peaks"][ref] if ref >= 0 else 0
            it = QListWidgetItem(f"#{h['number']:>3}  {h['status']:<9} {pk:8.4g} {run.channels[ref].units if ref >= 0 else ''}"
                                 + (f"  — {h['reason']}" if h["reason"] else ""))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if h["included"] else Qt.Unchecked)
            it.setBackground(QBrush(QColor(*STATUS_COLORS.get(h["status"], (127, 127, 127)), 50)))
            it.setToolTip(h["checks"])
            it.setData(Qt.UserRole, h["group"])
            self.hits.addItem(it)
        self.hits.blockSignals(False)
        if self.hits.count():
            self.hits.setCurrentRow(0)
        self.src.setCurrentIndex(0 if hits else 1)
        self._reset_windows(draw=False)
        names = [f"{c.name}  [{c.units}]" for c in run.channels]
        for combo, default in ((self.ch_a, max(ref, 0)), (self.ch_b, 1 if ref == 0 else 0), (self.sp_ch, max(ref, 0))):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            combo.setCurrentIndex(min(default, combo.count() - 1))
            combo.blockSignals(False)
        self._fill_resp()

    def _tick(self, pred):
        hits = load_hits(self.run) if self.run else []
        self.hits.blockSignals(True)
        for i, h in enumerate(hits):
            self.hits.item(i).setCheckState(Qt.Checked if pred(h) else Qt.Unchecked)
        self.hits.blockSignals(False)
        self._hits_changed()

    def _ticked(self) -> List[str]:
        return [self.hits.item(i).data(Qt.UserRole) for i in range(self.hits.count())
                if self.hits.item(i).checkState() == Qt.Checked]

    def _hits_changed(self, *_):
        self._res = None
        self._draw()

    # ------------------------------------------------------------------ windows
    def _reset_windows(self, *_, draw=True):
        if not self.run or not self.run.ok:
            return
        a = self.run.cfg.acq
        self.rp_fwin.setCurrentText(a.force_window)
        self.rp_fw.setValue(a.force_width_ms)
        self.rp_ft.setValue(a.force_taper_ms)
        self.rp_rwin.setCurrentText(a.resp_window)
        self.rp_end.setValue(a.exp_end_pct)
        self._acq = None
        self._res = None
        self.rp_info.setText("Windows as measured.")
        if draw:
            self._draw()

    def _apply_windows(self):
        if not self.run or not self.run.ok:
            return
        a = copy.deepcopy(self.run.cfg.acq)
        a.force_window, a.force_width_ms, a.force_taper_ms = (self.rp_fwin.currentText(), self.rp_fw.value(),
                                                              self.rp_ft.value())
        a.resp_window, a.exp_end_pct = self.rp_rwin.currentText(), self.rp_end.value()
        self._acq = a
        self._res = None
        hits = load_hits(self.run)
        if hits:
            n, pre = hits[0]["data"].shape[1], hits[0]["pre"]
            w = Windows.make(a, n, self.run.fs, pre)
            self.rp_info.setText("Re-processed windows" + (f": τ = {w.tau * 1000:.1f} ms" if np.isfinite(w.tau) else "")
                                 + ". Only 'Recomputed' FRFs and the spectra use them.")
        self.src.setCurrentIndex(0)
        self._draw()

    def _acq_now(self):
        return self._acq or self.run.cfg.acq

    # ------------------------------------------------------------------ FRF
    def _fill_resp(self, *_):
        run = self.run
        if not run or not run.ok:
            return
        cur = self.resp.currentText()
        if self.src.currentText() == SRC_STORED:
            names = list(frf_table(run, prefer_tdms=False).keys())
        else:
            names = [c.name for i, c in enumerate(run.channels) if i != run.ref_i]
        self.resp.blockSignals(True)
        self.resp.clear()
        self.resp.addItems(names)
        if cur in names:
            self.resp.setCurrentText(cur)
        self.resp.blockSignals(False)
        self._draw()

    def _draw(self, *_):
        if not self.run or not self.run.ok:
            return
        t = self.tabs.currentIndex()
        try:
            if t == 0:
                self._draw_frf()
            elif t == 1:
                self._draw_time()
            elif t == 2:
                self._draw_spec()
        except Exception as e:
            QMessageBox.warning(self, "Data Viewer", f"{type(e).__name__}: {e}")

    def _draw_frf(self):
        run = self.run
        name = self.resp.currentText()
        if not name:
            self.frf.clear()
            return
        if self.src.currentText() == SRC_STORED:
            d = frf_table(run, prefer_tdms=False).get(name)
            if d is None:
                self.frf.clear()
                self.frf_info.setText("not in the UFF files")
                return
            curves = {k: d[k] if self.est_cb[k].isChecked() else None for k in ("H1", "H2", "Hv")}
            self.frf.set_frf(d["freqs"], curves, d["q"], f"|H|  [{d['units']}]")
            missing = [k for k in ("H1", "H2", "Hv") if d[k] is None]
            self.frf_info.setText("stored" + (f" (no {', '.join(missing)} file)" if missing else ""))
            return
        if self._res is None:
            self._res = recompute(run, self._ticked(), self._acq_now())
        res = self._res
        if res is None:
            self.frf.clear()
            self.frf_info.setText("tick at least one hit")
            return
        names = [run.channels[i].name for i in res.resp_i]
        if name not in names:
            return
        j = names.index(name)
        c, r = run.channels[res.resp_i[j]], run.channels[run.ref_i]
        curves = {k: getattr(res, k)[j][1:] if self.est_cb[k].isChecked() else None for k in ("H1", "H2", "Hv")}
        self.frf.set_frf(res.freqs[1:], curves, res.coherence[j][1:], f"|H|  [{c.units}/{r.units}]")
        self.frf_info.setText(f"{res.n_avg} hit(s)")

    # ------------------------------------------------------------------ time
    def _draw_time(self):
        run = self.run
        hits = load_hits(run)
        for p in (self.p_ta, self.p_tb):
            p.clear()
        if not hits:
            self.p_ta.setTitle("No time data saved for this run (Hits → TDMS was off)")
            return
        ticked = set(self._ticked())
        cur = self.hits.currentRow()
        a, b = self.ch_a.currentIndex(), self.ch_b.currentIndex()
        n, pre = hits[0]["data"].shape[1], hits[0]["pre"]
        t = (np.arange(n) - pre) / run.fs
        for k, h in enumerate(hits):
            if h["group"] not in ticked or k == cur:
                continue
            col = QColor(*STATUS_COLORS.get(h["status"], (127, 127, 127)), 70)
            self.p_ta.plot(t, h["data"][a], pen=pg.mkPen(col, width=1))
            self.p_tb.plot(t, h["data"][b], pen=pg.mkPen(col, width=1))
        if 0 <= cur < len(hits):
            h = hits[cur]
            self.p_ta.plot(t, h["data"][a], pen=pg.mkPen(COLOR_A, width=2))
            self.p_tb.plot(t, h["data"][b], pen=pg.mkPen(COLOR_B, width=1.5))
            if self.t_trig.isChecked() and h["trigger"] is not None:
                tr = h["trigger"]
                sc = np.max(np.abs(h["data"][a])) / (np.max(np.abs(tr)) + 1e-30)
                self.p_ta.plot(t, tr * sc, pen=pg.mkPen("#8e44ad", width=1.5), stepMode=None)
            self.p_ta.setTitle(f"Hit {h['number']} ({h['status']}) bold; other ticked hits faint")
        if self.t_win.isChecked():
            w = Windows.make(self._acq_now(), n, run.fs, pre)
            for p, i in ((self.p_ta, a), (self.p_tb, b)):
                pk = max([np.max(np.abs(h["data"][i])) for h in hits] + [1e-30])
                win = w.w_force_only * w.w_exp if i == run.ref_i else w.w_exp
                p.plot(t, win * pk, pen=pg.mkPen("#1e8449", width=1.5, style=Qt.DashLine))
        for p, i in ((self.p_ta, a), (self.p_tb, b)):
            c = run.channels[i]
            p.setLabel("left", c.name, c.units)

    # ------------------------------------------------------------------ spectra
    def _draw_spec(self):
        run = self.run
        self.p_sp.clear()
        hits = load_hits(run)
        if not hits:
            return
        i = self.sp_ch.currentIndex()
        ticked = set(self._ticked())
        cur = self.hits.currentRow()
        n, pre = hits[0]["data"].shape[1], hits[0]["pre"]
        f = np.arange(n // 2 + 1)[1:] * run.fs / n
        win = Windows.make(self._acq_now(), n, run.fs, pre) if self.sp_win.isChecked() else None
        db = self.sp_db.isChecked()
        self.p_sp.setLogMode(y=not db)

        def spec(h):
            X = hit_spectra(h["data"], run.ref_i, win, run.fs)[i] if win else np.fft.rfft(h["data"][i]) / run.fs
            return np.abs(X[1:])

        def y(a):
            return 20 * np.log10(a + 1e-30) if db else a + 1e-30

        acc = []
        for k, h in enumerate(hits):
            if h["group"] not in ticked:
                continue
            a = spec(h)
            acc.append(a ** 2)
            if k != cur:
                col = QColor(*STATUS_COLORS.get(h["status"], (127, 127, 127)), 70)
                self.p_sp.plot(f, y(a), pen=pg.mkPen(col, width=1))
        if 0 <= cur < len(hits):
            self.p_sp.plot(f, y(spec(hits[cur])), pen=pg.mkPen(COLOR_A, width=2))
        if acc:
            self.p_sp.plot(f, y(np.sqrt(np.mean(acc, axis=0))), pen=pg.mkPen("k", width=2))
        c = run.channels[i]
        self.p_sp.setLabel("left", f"|{c.name}|" + (" (dB re 1)" if db else ""), "" if db else f"{c.units}·s")

    # ------------------------------------------------------------------ export
    def _save_reprocessed(self):
        run = self.run
        if not run or not run.ok or not self._ticked():
            QMessageBox.information(self, "Save", "Load a run with saved hits and tick some hits first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Folder for the re-processed UFF files", run.folder)
        if not d:
            return
        from ..export import write_frf_uff
        res = recompute(run, self._ticked(), self._acq_now())
        resp = [run.channels[i] for i in res.resp_i]
        ref = run.channels[run.ref_i]
        base = os.path.join(d, run.cfg.save.basename + "_reprocessed")
        info = f"Re-processed: {res.n_avg} hits of {run.name}"
        files = [write_frf_uff(f"{base}_{k}.uff", k, res.freqs[1:],
                               (res.coherence if k == "Coherence" else getattr(res, k))[:, 1:], resp, ref, info)
                 for k in ("H1", "H2", "Hv", "Coherence")]
        QMessageBox.information(self, "Saved", "Written:\n" + "\n".join(os.path.basename(f) for f in files))

    def _img(self, what: str) -> str:
        return f"{self.run.name if self.run else 'run'}_{what}.png"
