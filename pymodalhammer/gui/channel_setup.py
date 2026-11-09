# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Channel Setup tab: one table row per input channel (same as the shaker app)."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMenu, QPushButton, QSpinBox,
                               QTableWidget, QToolButton, QVBoxLayout, QWidget)

from ..config import (BRIDGE_TYPES, CUSTOM_TYPE, DIRECTIONS, SENSOR_TYPES, TYPE_DEFAULTS, UNITS_BY_TYPE,
                      ChannelConfig, sensitivity_units)

COLS = ["On", "Physical channel", "Name", "Node", "Dir", "Sensor type", "Sensitivity",
        "Sens. units", "Units", "Range ±", "IEPE (mA)", "Coupling", "Reference",
        "Bridge", "Gage factor", "Excit. (V)", "R nom (Ω)", "Poisson"]
C = {name: i for i, name in enumerate(COLS)}
STRAIN_COLS = ["Bridge", "Gage factor", "Excit. (V)", "R nom (Ω)", "Poisson"]

TIPS = {
    "On": "Include this channel in the measurement",
    "Physical channel": "DAQmx physical channel, e.g. cDAQ1Mod1/ai0 (pick from list or type)",
    "Node": "Geometry node number written to the UFF file",
    "Dir": "Measurement direction written to the UFF file",
    "Sensor type": "Pressure (IEPE): IEPE pressure transducer, sensitivity in mV per pressure unit.\n"
                   "Custom: any other sensor that outputs a voltage — type its name, pick or type its unit, "
                   "and give the sensitivity in mV per unit (set IEPE if it needs excitation).",
    "Sensitivity": "Sensor sensitivity from its calibration sheet (mV per unit, as shown in 'Sens. units')",
    "Units": "Engineering unit. Voltage and Custom: pick one or type your own (e.g. mm, µm, kPa)",
    "Range ±": "Expected maximum level in engineering units (sets the input range)",
    "IEPE (mA)": "IEPE excitation current. 0 = off. NI 9234/9250 support 2 mA; 9232 supports 2/4 mA",
    "Coupling": "AC for IEPE sensors; Default leaves the module's default",
    "Reference": "The impact hammer's force sensor. FRFs are computed as response / hammer force",
}


def _dspin(lo, hi, dec, val, step=None):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setValue(val)
    w.setFrame(False)
    if step:
        w.setSingleStep(step)
    w.setButtonSymbols(QDoubleSpinBox.NoButtons)
    return w


def _combo(items, editable=False):
    w = QComboBox()
    w.addItems(items)
    w.setEditable(editable)
    w.setFrame(False)
    return w


class ChannelRow:
    """Holds the editor widgets of one table row."""

    def __init__(self, table: "ChannelTable", ai_names: List[str]):
        self.table = table
        self.w = {}
        self.w["On"] = QCheckBox()
        self.w["Physical channel"] = _combo(ai_names, editable=True)
        self.w["Name"] = QLineEdit()
        self.w["Name"].setFrame(False)
        self.w["Node"] = QSpinBox()
        self.w["Node"].setRange(0, 999999)
        self.w["Node"].setFrame(False)
        self.w["Node"].setButtonSymbols(QSpinBox.NoButtons)
        self.w["Dir"] = _combo(DIRECTIONS)
        self.w["Sensor type"] = _combo(SENSOR_TYPES)
        self.w["Custom name"] = QLineEdit()
        self.w["Custom name"].setFrame(False)
        self.w["Custom name"].setPlaceholderText("type name, e.g. Laser displacement")
        self.w["Custom name"].setToolTip("Name of the custom sensor type (saved with the data)")
        self.w["Custom name"].setVisible(False)
        self.w["Custom name"].textChanged.connect(lambda _: table.changed.emit())
        tb = QWidget()
        tl = QHBoxLayout(tb)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)
        tl.addWidget(self.w["Sensor type"])
        tl.addWidget(self.w["Custom name"], 1)
        self.w["Sensor type_box"] = tb
        self.w["Sensitivity"] = _dspin(0, 1e6, 4, 100.0)
        self.w["Sens. units"] = QLabel()
        self.w["Sens. units"].setContentsMargins(6, 0, 6, 0)
        self.w["Units"] = _combo(["g"], editable=False)
        self.w["Range ±"] = _dspin(0, 1e9, 6, 50.0)
        self.w["IEPE (mA)"] = _combo(["0", "2", "4"])
        self.w["Coupling"] = _combo(["Default", "AC", "DC"])
        self.w["Reference"] = QCheckBox()
        self.w["Bridge"] = _combo(BRIDGE_TYPES)
        self.w["Gage factor"] = _dspin(0, 100, 3, 2.0)
        self.w["Excit. (V)"] = _dspin(0, 20, 3, 2.5)
        self.w["R nom (Ω)"] = _dspin(0, 1e5, 1, 350.0)
        self.w["Poisson"] = _dspin(0, 0.5, 3, 0.3)
        for k in ("On", "Reference"):   # centre the check boxes
            box = QWidget()
            lay = QHBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setAlignment(Qt.AlignCenter)
            lay.addWidget(self.w[k])
            self.w[k + "_box"] = box
        self.w["Sensor type"].currentTextChanged.connect(self._type_changed)
        self.w["Units"].currentTextChanged.connect(self._update_sens_units)
        self.w["Units"].editTextChanged.connect(self._update_sens_units)
        self.w["Reference"].toggled.connect(self._ref_toggled)
        self.w["On"].toggled.connect(lambda _: table.changed.emit())
        self._loading = False

    def widgets_in_order(self):
        for name in COLS:
            yield name, self.w.get(name + "_box", self.w[name])

    # ------------------------------------------------------------------ behaviour
    def _type_changed(self, t: str):
        units = self.w["Units"]
        units.blockSignals(True)
        units.clear()
        units.addItems(UNITS_BY_TYPE.get(t, ["EU"]))
        units.setEditable(t in ("Voltage", CUSTOM_TYPE))
        units.blockSignals(False)
        custom = t == CUSTOM_TYPE
        self.w["Custom name"].setVisible(custom)
        if custom:
            tw = getattr(self.table, "table", None)
            if tw is not None and tw.columnWidth(C["Sensor type"]) < 330:
                tw.setColumnWidth(C["Sensor type"], 330)
        if custom and not self._loading:
            self.w["Custom name"].setFocus()
        if not self._loading:
            sens, rng, iepe, coup = TYPE_DEFAULTS.get(t, (100.0, 10.0, 0.0, "Default"))
            self.w["Sensitivity"].setValue(sens)
            self.w["Range ±"].setValue(rng)
            self.w["IEPE (mA)"].setCurrentText(str(int(iepe)))
            self.w["Coupling"].setCurrentText(coup)
        strain = t.startswith("Strain")
        self.w["Range ±"].setDecimals(6 if strain else 3)
        for k in STRAIN_COLS:
            self.w[k].setEnabled(strain)
        self.w["Sensitivity"].setEnabled(not strain)
        self.w["IEPE (mA)"].setEnabled(not strain)
        self._update_sens_units()
        self.table.changed.emit()

    def _update_sens_units(self, *_):
        self.w["Sens. units"].setText(
            sensitivity_units(self.w["Sensor type"].currentText(), self.w["Units"].currentText()))

    def _ref_toggled(self, on: bool):
        if on and not self._loading:
            self.table.set_reference(self)
        self.table.changed.emit()

    # ------------------------------------------------------------------ data
    def set(self, c: ChannelConfig):
        self._loading = True
        w = self.w
        w["On"].setChecked(c.enabled)
        w["Physical channel"].setCurrentText(c.physical)
        w["Name"].setText(c.name)
        w["Node"].setValue(c.node)
        w["Dir"].setCurrentText(c.direction)
        w["Sensor type"].setCurrentText(c.sensor_type if c.sensor_type in SENSOR_TYPES else CUSTOM_TYPE)
        w["Custom name"].setText(c.custom_type if c.sensor_type == CUSTOM_TYPE else
                                 (c.sensor_type if c.sensor_type not in SENSOR_TYPES else ""))
        self._type_changed(w["Sensor type"].currentText())
        w["Sensitivity"].setValue(c.sensitivity)
        w["Units"].setCurrentText(c.units)
        w["Range ±"].setValue(c.max_range)
        w["IEPE (mA)"].setCurrentText(str(int(c.iepe_ma)))
        w["Coupling"].setCurrentText(c.coupling)
        w["Reference"].setChecked(c.is_reference)
        w["Bridge"].setCurrentText(c.bridge)
        w["Gage factor"].setValue(c.gage_factor)
        w["Excit. (V)"].setValue(c.excit_v)
        w["R nom (Ω)"].setValue(c.nominal_r)
        w["Poisson"].setValue(c.poisson)
        self._update_sens_units()
        self._loading = False

    def get(self) -> ChannelConfig:
        w = self.w
        return ChannelConfig(
            enabled=w["On"].isChecked(), physical=w["Physical channel"].currentText().strip(),
            name=w["Name"].text().strip() or "Ch", node=w["Node"].value(), direction=w["Dir"].currentText(),
            sensor_type=w["Sensor type"].currentText(), sensitivity=w["Sensitivity"].value(),
            units=w["Units"].currentText().strip() or "EU", max_range=w["Range ±"].value(),
            iepe_ma=float(w["IEPE (mA)"].currentText()), coupling=w["Coupling"].currentText(),
            is_reference=w["Reference"].isChecked(), bridge=w["Bridge"].currentText(),
            gage_factor=w["Gage factor"].value(), excit_v=w["Excit. (V)"].value(),
            nominal_r=w["R nom (Ω)"].value(), poisson=w["Poisson"].value(),
            custom_type=w["Custom name"].text().strip() if w["Sensor type"].currentText() == CUSTOM_TYPE else "")


class ChannelTable(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: List[ChannelRow] = []
        self.ai_names: List[str] = []

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        for name, tip in TIPS.items():
            self.table.horizontalHeaderItem(C[name]).setToolTip(tip)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setDefaultSectionSize(28)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        widths = {"On": 34, "Physical channel": 150, "Name": 120, "Node": 60, "Dir": 55,
                  "Sensor type": 170, "Sensitivity": 85, "Sens. units": 80, "Units": 70,
                  "Range ±": 75, "IEPE (mA)": 70, "Coupling": 80, "Reference": 70, "Bridge": 130}
        for k, v in widths.items():
            self.table.setColumnWidth(C[k], v)

        title = QLabel("<b>Input channels</b> — one row per sensor. Tick <i>Reference</i> on the "
                       "<b>impact hammer</b>'s force sensor; every other enabled channel is a response. "
                       "In roving-hammer mode the hammer's node and direction are set per point on the "
                       "Measurement tab.")
        title.setWordWrap(True)

        self.btn_add = QPushButton("＋ Add channel")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_del = QPushButton("Remove")
        self.btn_up = QPushButton("▲")
        self.btn_down = QPushButton("▼")
        self.btn_fill = QToolButton()
        self.btn_fill.setText("Bulk edit ▾")
        self.btn_fill.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self)
        m.addAction("Number nodes 1, 2, 3 … (selected / all)", self._number_nodes)
        m.addAction("Copy sensor settings of the current row to selected rows", self._copy_settings)
        m.addAction("Add every AI channel of a module…", self._add_module)
        self.btn_fill.setMenu(m)
        self.null_bridges = QCheckBox("Null strain-bridge offsets before each run")
        self.null_bridges.setChecked(True)
        self.summary = QLabel()

        self.btn_add.clicked.connect(lambda: self.add_row())
        self.btn_dup.clicked.connect(self._duplicate)
        self.btn_del.clicked.connect(self._remove)
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))
        self.changed.connect(self._update_summary)

        bar = QHBoxLayout()
        for b in (self.btn_add, self.btn_dup, self.btn_del, self.btn_up, self.btn_down, self.btn_fill):
            bar.addWidget(b)
        bar.addStretch()
        bar.addWidget(self.null_bridges)

        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addLayout(bar)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.summary)

    # ------------------------------------------------------------------ rows
    def add_row(self, c: ChannelConfig = None, index: int = None):
        if c is None:
            used = {r.get().physical for r in self.rows}
            free = [n for n in self.ai_names if n not in used]
            last = self.rows[-1].get() if self.rows else ChannelConfig()
            c = ChannelConfig(**{**last.__dict__, "is_reference": False})
            c.physical = free[0] if free else ""
            c.name = f"Ch{len(self.rows)}"
            c.node = last.node + 1 if self.rows else 1
        row = ChannelRow(self, self.ai_names)
        i = len(self.rows) if index is None else index
        self.table.insertRow(i)
        self.rows.insert(i, row)
        for name, wid in row.widgets_in_order():
            self.table.setCellWidget(i, C[name], wid)
        row.set(c)
        self.changed.emit()
        return row

    def _selected(self) -> List[int]:
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()}) or \
            ([self.table.currentRow()] if self.table.currentRow() >= 0 else [])

    def _duplicate(self):
        for i in reversed(self._selected()):
            c = self.rows[i].get()
            c.is_reference = False
            c.name += "_copy"
            self.add_row(c, i + 1)

    def _remove(self):
        for i in reversed(self._selected()):
            self.table.removeRow(i)
            self.rows.pop(i)
        self.changed.emit()

    def _move(self, step: int):
        sel = self._selected()
        if len(sel) != 1:
            return
        i, j = sel[0], sel[0] + step
        if not 0 <= j < len(self.rows):
            return
        cfgs = self.get_channels()
        cfgs[i], cfgs[j] = cfgs[j], cfgs[i]
        self.set_channels(cfgs)
        self.table.selectRow(j)

    def _number_nodes(self):
        idx = self._selected()
        if len(idx) < 2:
            idx = range(len(self.rows))
        for n, i in enumerate(idx, start=1):
            self.rows[i].w["Node"].setValue(n)

    def _copy_settings(self):
        cur = self.table.currentRow()
        if cur < 0:
            return
        src = self.rows[cur].get()
        for i in self._selected():
            if i == cur:
                continue
            c = self.rows[i].get()
            for k in ("sensor_type", "custom_type", "sensitivity", "units", "max_range", "iepe_ma", "coupling",
                      "bridge", "gage_factor", "excit_v", "nominal_r", "poisson"):
                setattr(c, k, getattr(src, k))
            self.rows[i].set(c)
        self.changed.emit()

    def _add_module(self):
        from PySide6.QtWidgets import QInputDialog
        mods = sorted({n.split("/")[0] for n in self.ai_names})
        if not mods:
            return
        mod, ok = QInputDialog.getItem(self, "Add module", "Module:", mods, 0, False)
        if ok:
            used = {r.get().physical for r in self.rows}
            for n in self.ai_names:
                if n.startswith(mod + "/") and n not in used:
                    self.add_row().w["Physical channel"].setCurrentText(n)

    def set_reference(self, row: ChannelRow):
        for r in self.rows:
            if r is not row and r.w["Reference"].isChecked():
                r.w["Reference"].blockSignals(True)
                r.w["Reference"].setChecked(False)
                r.w["Reference"].blockSignals(False)

    # ------------------------------------------------------------------ data
    def set_ai_names(self, names: List[str]):
        self.ai_names = list(names)
        for r in self.rows:
            cb = r.w["Physical channel"]
            cur = cb.currentText()
            cb.clear()
            cb.addItems(self.ai_names)
            cb.setCurrentText(cur)

    def set_ranges(self, ranges: dict):
        """{channel name: ± range} (from Probe mode's suggestion)."""
        for r in self.rows:
            n = r.w["Name"].text().strip()
            if n in ranges:
                r.w["Range ±"].setValue(float(ranges[n]))
        self.changed.emit()

    def get_channels(self) -> List[ChannelConfig]:
        return [r.get() for r in self.rows]

    def set_channels(self, chans: List[ChannelConfig]):
        self.table.setRowCount(0)
        self.rows.clear()
        for c in chans:
            self.add_row(c)
        self.changed.emit()

    def _update_summary(self):
        ch = [r.get() for r in self.rows]
        en = [c for c in ch if c.enabled]
        ref = [c for c in en if c.is_reference]
        txt = f"{len(en)} of {len(ch)} channels enabled · "
        if len(ref) == 1:
            txt += (f"hammer (reference): <b>{ref[0].name}</b> (node {ref[0].node}{ref[0].direction}) · "
                    + (f"{len(en) - 1} response channel(s) → {len(en) - 1} FRFs" if len(en) > 1 else
                       "no response channels → <b>time data only</b> (hits recorded and saved, no FRFs)"))
        else:
            txt += "<span style='color:#c0392b'><b>tick Reference on the hammer (exactly one enabled channel)</b></span>"
        self.summary.setText(txt)
