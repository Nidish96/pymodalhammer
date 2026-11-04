# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Shared plotting helpers: a two-channel plot with independent left/right y-axes, block
decorations (separators, labels, shading), the FRF stack, and image export of any widget."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QMarginsF, QPoint, QRect, QSize, QSizeF, Qt
from PySide6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter, QPixmap
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

COLOR_A = "#2b7bb9"
COLOR_B = "#d35400"
EST_COLORS = {"H1": "#2b7bb9", "H2": "#e67e22", "Hv": "#27ae60"}


class DualAxisPlot(pg.PlotWidget):
    """Curve A on the left axis, optional curve B on its own right axis (different units)."""

    def __init__(self, parent=None, downsample: bool = False, **kw):
        super().__init__(parent, **kw)
        pi = self.getPlotItem()
        self.showGrid(x=True, y=True, alpha=0.3)
        for ax in ("left", "bottom", "right"):
            pi.getAxis(ax).enableAutoSIPrefix(False)     # keep 'g' as g, not 'mg'
        self.vb2 = pg.ViewBox()
        pi.scene().addItem(self.vb2)
        pi.getAxis("right").linkToView(self.vb2)
        self.vb2.setXLink(pi)
        pi.vb.sigResized.connect(self._sync)
        self.curve_a = pi.plot(pen=pg.mkPen(COLOR_A, width=1))
        self.curve_b = pg.PlotDataItem(pen=pg.mkPen(COLOR_B, width=1))
        self.vb2.addItem(self.curve_b)
        if downsample:
            for c in (self.curve_a, self.curve_b):
                c.setDownsampling(auto=True, method="peak")
                c.setClipToView(True)
        pi.hideAxis("right")
        self._deco: List = []

    def _sync(self):
        pi = self.getPlotItem()
        self.vb2.setGeometry(pi.vb.sceneBoundingRect())
        self.vb2.linkedViewChanged(pi.vb, self.vb2.XAxis)

    def set_log_y(self, log: bool):
        """Logarithmic y for both curves and both axes."""
        pi = self.getPlotItem()
        pi.setLogMode(y=log)
        self.curve_b.setLogMode(False, log)
        pi.getAxis("right").setLogMode(False, log)
        self.vb2.enableAutoRange(axis=pg.ViewBox.YAxis)

    def set_curves(self, x, ya, label_a="", unit_a="", yb=None, label_b="", unit_b=""):
        pi = self.getPlotItem()
        self.curve_a.setData(x, ya)
        pi.setLabel("left", label_a, unit_a, color=COLOR_A)
        if yb is None:
            self.curve_b.setData([], [])
            pi.hideAxis("right")
        else:
            self.curve_b.setData(x, yb)
            pi.showAxis("right")
            pi.setLabel("right", label_b, unit_b, color=COLOR_B)
            self.vb2.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._sync()

    # ---------------------------------------------------------------- block decorations
    def clear_decorations(self):
        for it in self._deco:
            self.removeItem(it)
        self._deco = []

    def add_separator(self, x: float):
        ln = pg.InfiniteLine(pos=x, angle=90, movable=False,
                             pen=pg.mkPen("#7f8c8d", width=1, style=Qt.DashLine))
        self.addItem(ln)
        self._deco.append(ln)

    def add_shading(self, a: float, b: float, rgba=(46, 204, 113, 40)):
        reg = pg.LinearRegionItem(values=(a, b), movable=False, brush=pg.mkBrush(*rgba),
                                  pen=pg.mkPen(None))
        reg.setZValue(-10)
        self.addItem(reg)
        self._deco.append(reg)

    def add_label(self, x: float, y: float, text: str, highlight: bool = False):
        color = "#1e8449" if highlight else "#2c3e50"
        weight = "bold" if highlight else "normal"
        lbl = pg.TextItem(html=f"<span style='color:{color}; font-size:8pt; font-weight:{weight}'>"
                               f"{text}</span>", anchor=(0.5, 0))
        lbl.setPos(x, y)
        self.addItem(lbl)
        self._deco.append(lbl)

    def decorate_blocks(self, starts: Sequence[float], length: float, texts: Sequence[str],
                        converged: Sequence[bool], y_top: float, selected: Optional[int] = None):
        """Dashed separators between blocks, a label per block, converged block(s) shaded green,
        optionally one selected block shaded blue."""
        self.clear_decorations()
        for i, (a, txt, conv) in enumerate(zip(starts, texts, converged)):
            if conv:
                self.add_shading(a, a + length)
            if selected is not None and i == selected and not conv:
                self.add_shading(a, a + length, (52, 152, 219, 40))
            self.add_separator(a)
            self.add_label(a + 0.5 * length, y_top, txt, conv)
        if len(starts):
            self.add_separator(starts[-1] + length)


class FRFStack(QWidget):
    """Coherence (top), FRF magnitude (middle), phase (bottom), x-linked, frequency in Hz."""

    def __init__(self, parent=None):
        from PySide6.QtWidgets import QSplitter, QVBoxLayout
        super().__init__(parent)
        self.p_coh, self.p_mag, self.p_ph = pg.PlotWidget(), pg.PlotWidget(), pg.PlotWidget()
        self.p_mag.setLogMode(y=True)
        self.p_mag.setLabel("left", "|H|")
        self.p_ph.setLabel("left", "Phase", "°")
        self.p_ph.setYRange(-180, 180)
        self.p_coh.setLabel("left", "Coherence γ²")
        self.p_coh.setYRange(0, 1.05)
        self.p_ph.setLabel("bottom", "Frequency", "Hz")
        for p in (self.p_coh, self.p_mag, self.p_ph):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            if p is not self.p_mag:       # log |H| axis keeps pyqtgraph's own tick labelling
                p.getAxis("left").enableAutoSIPrefix(False)
        self.p_ph.setXLink(self.p_mag)
        self.p_coh.setXLink(self.p_mag)
        self.legend = self.p_mag.addLegend(offset=(-10, 10))
        self.mag = {k: self.p_mag.plot(pen=pg.mkPen(c, width=1.5), name=k) for k, c in EST_COLORS.items()}
        self.ph = {k: self.p_ph.plot(pen=pg.mkPen(c, width=1.2)) for k, c in EST_COLORS.items()}
        self.coh = self.p_coh.plot(pen=pg.mkPen("#8e44ad", width=1.5))
        self.band_lines = []
        for p in (self.p_coh, self.p_mag, self.p_ph):
            for _ in range(2):
                ln = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#95a5a6", style=Qt.DashLine))
                ln.setVisible(False)
                p.addItem(ln)
                self.band_lines.append(ln)
        split = QSplitter(Qt.Vertical)
        for p in (self.p_coh, self.p_mag, self.p_ph):
            split.addWidget(p)
        split.setSizes([140, 300, 160])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

    def set_band(self, band):
        for i, ln in enumerate(self.band_lines):
            if band is None:
                ln.setVisible(False)
            else:
                ln.setPos(band[i % 2])
                ln.setVisible(True)

    def set_frf(self, f, curves: dict, coh=None, mag_label="|H|"):
        """curves: {'H1': complex array or None, ...}; f must exclude DC (log axis)."""
        for k in EST_COLORS:
            H = curves.get(k)
            if H is None:
                self.mag[k].setData([], [])
                self.ph[k].setData([], [])
            else:
                self.mag[k].setData(f, np.abs(H) + 1e-30)
                self.ph[k].setData(f, np.degrees(np.angle(H)))
        if coh is None:
            self.coh.setData([], [])
        else:
            self.coh.setData(f, coh)
        self.p_mag.setLabel("left", mag_label)

    def show_aux(self, on: bool):
        """Show/hide the quality (coherence/η) and phase plots, e.g. for amplitude spectra."""
        self.p_coh.setVisible(on)
        self.p_ph.setVisible(on)
        self.p_mag.setLabel("bottom", "" if on else "Frequency", "" if on else "Hz")

    def set_quality_label(self, text: str):
        """Top plot: 'Coherence γ²' (random) or 'Fundamental fraction η' (stepped sine)."""
        self.p_coh.setLabel("left", text)

    def set_points(self, on: bool):
        """Show markers (stepped sine: one value per measured frequency)."""
        for k, c in list(self.mag.items()) + list(self.ph.items()):
            c.setSymbol("o" if on else None)
            c.setSymbolSize(5)
            c.setSymbolBrush(EST_COLORS[k])
            c.setSymbolPen(None)
        self.coh.setSymbol("o" if on else None)
        self.coh.setSymbolSize(5)
        self.coh.setSymbolBrush("#8e44ad")
        self.coh.setSymbolPen(None)

    def clear(self):
        self.set_frf([], {}, None)


# ----------------------------------------------------------------------------- image export
def export_widget_image(widget: QWidget, path: str, scale: float = 2.0) -> None:
    """Save what `widget` shows as PNG/JPG (scaled), SVG (vector) or PDF (vector)."""
    ext = os.path.splitext(path)[1].lower()
    size = widget.size()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        pm = QPixmap(QSize(int(size.width() * scale), int(size.height() * scale)))
        pm.fill(Qt.white)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.scale(scale, scale)
        widget.render(p, QPoint(0, 0))
        p.end()
        if not pm.save(path):
            raise IOError(f"Could not write {path}")
    elif ext == ".svg":
        from PySide6.QtSvg import QSvgGenerator
        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(size)
        gen.setViewBox(QRect(0, 0, size.width(), size.height()))
        gen.setTitle(os.path.basename(path))
        p = QPainter(gen)
        widget.render(p, QPoint(0, 0))
        p.end()
    elif ext == ".pdf":
        pdf = QPdfWriter(path)
        pdf.setResolution(96)
        pdf.setPageSize(QPageSize(QSizeF(size.width() * 0.75, size.height() * 0.75), QPageSize.Point))
        pdf.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Point)
        p = QPainter(pdf)
        widget.render(p, QPoint(0, 0))
        p.end()
    else:
        raise ValueError(f"Unsupported image type '{ext}'. Use .png, .jpg, .svg or .pdf")


def export_dialog(parent: QWidget, widget: QWidget, default_name: str) -> Optional[str]:
    """Ask for a file name and export `widget`. Returns the path written, or None."""
    path, flt = QFileDialog.getSaveFileName(
        parent, "Export image", default_name,
        "PNG image, 2× resolution (*.png);;SVG vector (*.svg);;PDF vector (*.pdf);;JPEG image (*.jpg)")
    if not path:
        return None
    if not os.path.splitext(path)[1]:
        path += {"PNG": ".png", "SVG": ".svg", "PDF": ".pdf", "JPE": ".jpg"}.get(flt[:3], ".png")
    try:
        export_widget_image(widget, path)
    except Exception as e:
        QMessageBox.critical(parent, "Export image", str(e))
        return None
    return path
