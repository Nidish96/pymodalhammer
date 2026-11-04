# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Reading saved measurement points (run folders) for the Data Viewer and Compare tabs."""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import AppConfig, ChannelConfig
from .processing import FRFResult, Windows, average, hit_spectra

UFF_FUNCS = ("H1", "H2", "Hv", "Coherence", "AutoSpectra", "time")


@dataclass
class Run:
    folder: str
    cfg: Optional[AppConfig] = None
    meta: Dict = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    channels: List[ChannelConfig] = field(default_factory=list)
    ref_i: int = -1
    fs: float = 0.0
    problems: List[str] = field(default_factory=list)
    _hits: Optional[List[Dict]] = None
    _uff: Dict[str, list] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.cfg is not None and not self.problems

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.folder))

    @property
    def point_label(self) -> str:
        if self.cfg is None:
            return ""
        s = self.cfg.save
        return f"P{s.point_node}{s.point_dir}" if s.roving else "fixed hammer"

    def summary_html(self) -> str:
        if self.cfg is None:
            return f"<b>{self.name}</b>: " + "; ".join(self.problems)
        a, m = self.cfg.acq, self.meta
        ref = self.channels[self.ref_i] if self.ref_i >= 0 else None
        hs = m.get("hits", {})
        hits_txt = ", ".join(f"{v} {k}" for k, v in hs.items()) or "—"
        tau = m.get("exp_tau_s")
        have = [k for k in UFF_FUNCS if k in self.files] + [k for k in ("hits", "hit_log", "trigger_log", "stream")
                                                            if k in self.files]
        return (f"<b>{self.name}</b> — saved {m.get('saved', '?')} · <b>{self.point_label}</b>"
                + (f" (hammer: {ref.name} at node {ref.node}{ref.direction})" if ref else "")
                + f"<br>fs = {self.fs:g} Hz, record {a.block_size} samples = {a.block_size / max(self.fs, 1):.3f} s, "
                  f"Δf = {self.fs / a.block_size:.4g} Hz, pre-trigger {a.pretrigger_pct:g} %, trigger "
                  f"{a.trig_level:g} on {a.trig_channel or (ref.name if ref else '?')}"
                + f"<br>{m.get('n_averages', '?')} averages · hits: {hits_txt} · windows: {a.force_window.split(' (')[0]}"
                  f" {a.force_width_ms:g} ms + {a.resp_window.split(' (')[0]}"
                + (f" (end {a.exp_end_pct:g} %, τ = {tau * 1000:.1f} ms)" if tau else "")
                + f"<br>Files: {', '.join(have) if have else 'none'}")

    # ---------------------------------------------------------------- UFF
    def uff_sets(self, func: str) -> list:
        if func not in self._uff:
            p = self.files.get(func)
            if not p:
                return []
            import pyuff
            sets = pyuff.UFF(p).read_sets()
            self._uff[func] = sets if isinstance(sets, list) else [sets]
        return self._uff[func]


def _uff_name(s: dict) -> str:
    parts = str(s.get("id1", "")).split()
    return parts[1] if len(parts) > 1 else str(s.get("rsp_ent_name", "")).strip()


def is_run(folder: str) -> bool:
    p = os.path.join(folder, "setup.json")
    if not os.path.isfile(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            return "acq" in json.load(f)
    except Exception:
        return False


def find_runs(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    if is_run(folder):
        return [folder]
    out = [os.path.join(folder, d) for d in sorted(os.listdir(folder))
           if os.path.isdir(os.path.join(folder, d)) and is_run(os.path.join(folder, d))]
    return out


def scan_run(folder: str) -> Run:
    run = Run(folder)
    p = os.path.join(folder, "setup.json")
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        run.meta = d.get("run", {})
        run.cfg = AppConfig.from_json(json.dumps(d))
    except Exception as e:
        run.problems.append(f"cannot read setup.json ({e})")
        return run
    base = os.path.join(folder, run.cfg.save.basename)
    for k in UFF_FUNCS:
        if os.path.isfile(f"{base}_{k}.uff"):
            run.files[k] = f"{base}_{k}.uff"
    for k, suf in (("hits", "_hits.tdms"), ("hit_log", "_hit_log.csv"), ("trigger_log", "_trigger_log.csv"),
                   ("stream", "_stream.tdms")):
        if os.path.isfile(base + suf):
            run.files[k] = base + suf
    if not run.files.get("hits") and not any(k in run.files for k in UFF_FUNCS):     # base name changed?
        for pth in glob.glob(os.path.join(folder, "*_hits.tdms")):
            run.files["hits"] = pth
    run.channels = run.cfg.labelled_channels()
    run.ref_i = run.cfg.reference_index()
    run.fs = float(run.meta.get("fs_actual") or run.cfg.acq.fs)
    return run


# ----------------------------------------------------------------------------- hits
def load_hits(run: Run) -> List[Dict]:
    """Every hit in <base>_hits.tdms: dict(group, number, status, included, reason, sample, data (n_ch, N),
    names, trigger (or None), pre, peaks)."""
    if run._hits is not None:
        return run._hits
    out = []
    p = run.files.get("hits")
    if p:
        from nptdms import TdmsFile
        f = TdmsFile.read(p)
        names = [c.name for c in run.channels]
        for g in f.groups():
            gp = g.properties
            if str(gp.get("status", "")) == "removed":       # deleted during the measurement (older files)
                continue
            have = {c.name for c in g.channels()}
            rows = [g[n][:] if n in have else np.zeros(len(g.channels()[0])) for n in names]
            out.append({"group": g.name, "number": int(gp.get("hit", 0)), "status": str(gp.get("status", "")),
                        "included": bool(gp.get("included", True)), "reason": str(gp.get("reason", "")),
                        "sample": int(gp.get("trigger_sample", 0)), "pre": int(gp.get("pre_samples", run.cfg.acq.pre_samples)),
                        "segment": int(gp.get("segment", 1)), "wall_time": str(gp.get("wall_time", "")),
                        "data": np.vstack(rows), "trigger": g["Trigger"][:] if "Trigger" in have else None,
                        "checks": str(gp.get("checks", ""))})
            out[-1]["peaks"] = np.max(np.abs(out[-1]["data"]), axis=1)
    run._hits = out
    return out


def recompute(run: Run, groups: List[str], acq=None, ref_i: Optional[int] = None) -> Optional[FRFResult]:
    """FRFs from the chosen hits of the TDMS file, with the run's windows (or `acq`'s)."""
    acq = acq or run.cfg.acq
    ref_i = run.ref_i if ref_i is None else ref_i
    hits = [h for h in load_hits(run) if h["group"] in groups]
    if not hits:
        return None
    n = hits[0]["data"].shape[1]
    pre = hits[0]["pre"]
    win = Windows.make(acq, n, run.fs, pre)
    freqs = np.arange(n // 2 + 1) * run.fs / n
    return average([hit_spectra(h["data"], ref_i, win, run.fs) for h in hits], ref_i, freqs)


def included_groups(run: Run) -> List[str]:
    return [h["group"] for h in load_hits(run) if h["included"]]


def frf_table(run: Run, prefer_tdms: bool = True) -> Dict[str, Dict]:
    """{response name: {freqs, H1, H2, Hv, q, units, ref}} from the time data (the hits that were
    averaged) or else the stored UFF files."""
    out: Dict[str, Dict] = {}
    ref = run.channels[run.ref_i] if run.ref_i >= 0 else None
    if prefer_tdms and run.files.get("hits") and ref is not None:
        res = recompute(run, included_groups(run))
        if res is not None:
            for j, i in enumerate(res.resp_i):
                c = run.channels[i]
                out[c.name] = {"freqs": res.freqs[1:], "H1": res.H1[j, 1:], "H2": res.H2[j, 1:],
                               "Hv": res.Hv[j, 1:], "q": res.coherence[j, 1:], "units": f"{c.units}/{ref.units}",
                               "ref": f"{ref.name} {ref.node}{ref.direction}", "source": "TDMS"}
            return out
    coh = {_uff_name(s): np.real(s["data"]) for s in run.uff_sets("Coherence")}
    for est in ("H1", "H2", "Hv"):
        for s in run.uff_sets(est):
            nm = _uff_name(s)
            d = out.setdefault(nm, {"freqs": np.asarray(s["x"], float), "H1": None, "H2": None, "Hv": None,
                                    "q": coh.get(nm), "units": f"{s.get('ordinate_axis_units_lab', '')}/"
                                                               f"{s.get('orddenom_axis_units_lab', '')}",
                                    "ref": f"{s.get('ref_ent_name', '')} {s.get('ref_node', '')}", "source": "UFF"})
            d[est] = np.asarray(s["data"])
    return out


def stored_frf(run: Run, func: str) -> Dict[str, Dict]:
    """{response: {freqs, data}} straight from one UFF file."""
    return {_uff_name(s): {"freqs": np.asarray(s["x"], float), "data": np.asarray(s["data"])}
            for s in run.uff_sets(func)}


def amplitude_spectra(run: Run, groups: Optional[List[str]] = None, windowed: bool = True) -> Dict[str, Dict]:
    """{channel: {freqs, amp (RMS over hits of |X(f)| [EU·s]), units}} of the chosen (default: averaged) hits."""
    groups = included_groups(run) if groups is None else groups
    hits = [h for h in load_hits(run) if h["group"] in groups]
    if not hits:
        return {}
    n = hits[0]["data"].shape[1]
    pre = hits[0]["pre"]
    win = Windows.make(run.cfg.acq, n, run.fs, pre) if windowed else None
    freqs = np.arange(n // 2 + 1) * run.fs / n
    if win is None:
        S = np.stack([np.fft.rfft(h["data"], axis=-1) / run.fs for h in hits])
    else:
        S = np.stack([hit_spectra(h["data"], run.ref_i, win, run.fs) for h in hits])
    amp = np.sqrt(np.mean(np.abs(S) ** 2, axis=0))
    return {c.name: {"freqs": freqs[1:], "amp": amp[i, 1:], "units": f"{c.units}·s"}
            for i, c in enumerate(run.channels)}


def trigger_rows(run: Run) -> List[Dict]:
    p = run.files.get("trigger_log")
    if not p:
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def time_records(run: Run) -> Dict[str, Dict]:
    """{channel: {t (s, 0 = trigger), hits: [{number, included, status, y, ref_peak}], units, win}} of the
    saved hits, for overlaying time data across runs. `win` is the run's window for that channel."""
    hits = load_hits(run)
    if not hits:
        return {}
    n, pre = hits[0]["data"].shape[1], hits[0]["pre"]
    t = (np.arange(n) - pre) / run.fs
    w = Windows.make(run.cfg.acq, n, run.fs, pre)
    out = {}
    for i, c in enumerate(run.channels):
        out[c.name] = {"t": t, "units": c.units, "win": w.force if i == run.ref_i else w.resp,
                       "ref_units": run.channels[run.ref_i].units if run.ref_i >= 0 else "",
                       "hits": [{"number": h["number"], "included": h["included"], "status": h["status"],
                                 "y": h["data"][i], "ref_peak": float(h["peaks"][run.ref_i]) if run.ref_i >= 0 else 1.0}
                                for h in hits]}
    return out
