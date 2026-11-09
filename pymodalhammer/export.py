# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""File export: UFF dataset 58 via pyuff (FRFs, coherence, auto-spectra, time responses), TDMS via
npTDMS (hit records, continuous stream) and CSV logs."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyuff
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

from .config import DIRECTION_CODE, AppConfig, ChannelConfig
from .processing import Hit, PointSession

# UFF58 "specific data type" codes
_SPEC_TYPE = {
    "Accelerometer (IEPE)": 12, "Force (IEPE)": 13, "Strain gauge (bridge)": 3,
    "Microphone (IEPE)": 15, "Pressure (IEPE)": 15, "Voltage": 1, "Custom": 1,
}
# UFF58 function types
FUNC = {"Time": 1, "Auto": 2, "H1": 4, "H2": 4, "Hv": 4, "Coherence": 6}


def _dataset(func_type, data, x, resp: ChannelConfig, ref: ChannelConfig, id1, id2, id3, rec_no,
             ordinate_units: str, denom: bool, spec_override: Optional[int] = None) -> dict:
    x = np.asarray(x, dtype=float)
    even = len(x) < 3 or np.allclose(np.diff(x), x[1] - x[0], rtol=1e-9, atol=1e-12)
    time_axis = func_type == 1
    return pyuff.prepare_58(
        binary=0, id1=id1[:80], id2=id2[:80], id3=id3[:80], id4="", id5="",
        func_type=func_type, ver_num=0, load_case_id=0,
        rsp_ent_name=resp.name[:10], rsp_node=int(resp.node), rsp_dir=DIRECTION_CODE[resp.direction],
        ref_ent_name=ref.name[:10], ref_node=int(ref.node), ref_dir=DIRECTION_CODE[ref.direction],
        ord_data_type=6 if np.iscomplexobj(data) else 4,
        num_pts=len(x), abscissa_spacing=1 if even else 0, abscissa_min=float(x[0]),
        abscissa_inc=float(x[1] - x[0]) if len(x) > 1 else 0.0,
        abscissa_spec_data_type=17 if time_axis else 18, abscissa_len_unit_exp=0, abscissa_force_unit_exp=0,
        abscissa_temp_unit_exp=0, abscissa_axis_units_lab="s" if time_axis else "Hz",
        ordinate_spec_data_type=spec_override if spec_override is not None else _SPEC_TYPE.get(resp.sensor_type, 1),
        ordinate_len_unit_exp=0, ordinate_force_unit_exp=0, ordinate_temp_unit_exp=0,
        ordinate_axis_units_lab=ordinate_units,
        orddenom_spec_data_type=_SPEC_TYPE.get(ref.sensor_type, 13) if denom else 0,
        orddenom_len_unit_exp=0, orddenom_force_unit_exp=0, orddenom_temp_unit_exp=0,
        orddenom_axis_units_lab=ref.units if denom else "",
        z_axis_spec_data_type=0, z_axis_len_unit_exp=0, z_axis_force_unit_exp=0,
        z_axis_temp_unit_exp=0, z_axis_axis_units_lab="", z_axis_value=float(rec_no),
        data=np.asarray(data), x=x)


def _write_sets(path: str, sets: list) -> str:
    if os.path.exists(path):
        os.remove(path)
    pyuff.UFF(path).write_sets(sets, mode="overwrite")
    return path


def write_frf_uff(path: str, func: str, freqs, data, responses: Sequence[ChannelConfig], ref: ChannelConfig,
                  info: str) -> str:
    """One dataset 58 per response. func: 'H1' | 'H2' | 'Hv' | 'Coherence'; data (n_resp, nf)."""
    stamp = _dt.datetime.now().strftime("%d-%b-%y %H:%M:%S")
    coh = func == "Coherence"
    sets = []
    for i, r in enumerate(responses):
        id1 = f"{func} {r.name} {r.node}{r.direction} / {ref.name} {ref.node}{ref.direction}"
        sets.append(_dataset(FUNC[func], np.real(data[i]) if coh else data[i], freqs, r, ref, id1, info, stamp,
                             i + 1, "" if coh else r.units, denom=not coh, spec_override=0 if coh else None))
    return _write_sets(path, sets)


def write_auto_uff(path: str, freqs, Gxx, Gyy, responses: Sequence[ChannelConfig], ref: ChannelConfig,
                   info: str) -> str:
    """Averaged auto-spectra |X|², |Y|² [EU²·s²] (UFF 58 type 2): the force first, then every response."""
    stamp = _dt.datetime.now().strftime("%d-%b-%y %H:%M:%S")
    sets = [_dataset(2, np.asarray(Gxx, float), freqs, ref, ref, f"Auto {ref.name} {ref.node}{ref.direction}",
                     info, stamp, 1, f"{ref.units}^2 s^2", denom=False)]
    for i, r in enumerate(responses):
        sets.append(_dataset(2, np.asarray(Gyy[i], float), freqs, r, r, f"Auto {r.name} {r.node}{r.direction}",
                             info, stamp, i + 2, f"{r.units}^2 s^2", denom=False))
    return _write_sets(path, sets)


def write_time_uff(path: str, hits: Sequence[Hit], channels: Sequence[ChannelConfig], ref: ChannelConfig,
                   info: str) -> str:
    """Hit records as time responses (UFF 58 type 1); t = 0 at the trigger point; one record per
    channel per hit, numbered by hit in the z-axis value."""
    stamp = _dt.datetime.now().strftime("%d-%b-%y %H:%M:%S")
    sets = []
    for h in hits:
        t = (np.arange(h.data.shape[1]) - h.pre) / h.fs
        for c, row in zip(channels, h.data):
            sets.append(_dataset(1, row, t, c, ref, f"Time {c.name} {c.node}{c.direction} hit {h.number}",
                                 info, stamp, h.number, c.units, denom=False))
    return _write_sets(path, sets)


# ----------------------------------------------------------------------------- TDMS
def _chan_props(c: ChannelConfig, fs: float, n: int, t0, offset: float = 0.0) -> Dict:
    return {"wf_increment": 1.0 / fs, "wf_start_offset": offset, "wf_samples": n, "wf_start_time": t0,
            "unit_string": c.units, "NI_ChannelName": c.name, "physical_channel": c.physical,
            "node": int(c.node), "direction": c.direction, "sensor_type": c.type_label,
            "sensitivity": float(c.sensitivity), "sensitivity_units": c.sens_units,
            "is_reference": bool(c.is_reference), "range": float(c.max_range)}


def root_properties(cfg: AppConfig, fs: float, extra: Optional[Dict] = None) -> Dict:
    a = cfg.acq
    p = {"name": cfg.save.basename,
         "description": "Impact test, time data only" if cfg.time_only else "Impact hammer modal test",
         "sample_rate_Hz": fs, "record_samples": a.block_size, "pretrigger_samples": a.pre_samples,
         "trigger_level": a.trig_level, "trigger_slope": a.trig_slope, "force_window": a.force_window,
         "force_width_ms": a.force_width_ms, "force_taper_ms": a.force_taper_ms, "response_window": a.resp_window,
         "exp_end_pct": a.exp_end_pct, "averages_requested": a.n_averages, "remove_offset": a.remove_offset,
         "point_node": int(cfg.save.point_node), "point_direction": cfg.save.point_dir,
         "point_mode": cfg.save.point_mode}
    t = cfg.trigger
    if t.enabled:
        p.update({"ext_trigger": t.physical, "ext_trigger_mode": t.mode, "ext_trigger_timing": t.timing,
                  "ext_trigger_pulse_samples": t.pulse_samples})
    p.update(extra or {})
    return p


def write_hits_tdms(path: str, cfg: AppConfig, fs: float, channels: Sequence[ChannelConfig],
                    hits: Sequence[Hit], extra_root: Optional[Dict] = None) -> str:
    """One group per hit (Hit_001 …): every channel (t = 0 at the trigger: wf_start_offset = −pre/fs),
    plus 'Trigger' if the external trigger was hardware-timed. Group properties: status, included,
    reason, checks, trigger sample, peaks, wall-clock time."""
    t0 = np.datetime64(_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None), "us")
    with TdmsWriter(path) as w:
        w.write_segment([RootObject(properties=root_properties(cfg, fs, extra_root))])
        for h in hits:
            g = f"Hit_{h.number:03d}"
            props = {"hit": h.number, "segment": h.segment, "status": h.status, "included": bool(h.included),
                     "reason": h.reason, "trigger_sample": int(h.sample), "trigger_time_s": h.sample / fs,
                     "wall_time": h.wall_time, "pre_samples": int(h.pre),
                     "checks": "; ".join(f"{c.name}: {'ok' if c.ok else 'FAIL'} ({c.message})" for c in h.checks),
                     "trigger_pulses": ",".join(str(p) for p in h.pulses)}
            for c, pk in zip(channels, h.peaks):
                props[f"peak_{c.name}"] = float(pk)
            objs = [GroupObject(g, properties=props)]
            for c, row in zip(channels, h.data):
                objs.append(ChannelObject(g, c.name, np.asarray(row, np.float64),
                                          properties=_chan_props(c, fs, row.size, t0, -h.pre / fs)))
            if h.trig_wave is not None:
                objs.append(ChannelObject(g, "Trigger", np.asarray(h.trig_wave, np.float64),
                                          properties={"wf_increment": 1.0 / fs, "wf_start_offset": -h.pre / fs,
                                                      "unit_string": "", "description":
                                                      "External trigger output (V for AO, 0/1 for DO)"}))
            w.write_segment(objs)
    return path


class TdmsStreamWriter:
    """The complete continuous record, appended read by read (valid even if the PC crashes)."""

    def __init__(self, path: str, cfg: AppConfig, fs: float, channels: Sequence[ChannelConfig], segment: int,
                 with_trigger: bool = False):
        self.path, self.channels, self.fs = path, list(channels), fs
        self.group = f"Segment_{segment:02d}"
        self.with_trigger = with_trigger
        new = not os.path.exists(path)
        self._w = TdmsWriter(path, mode="w" if new else "a")
        self._w.__enter__()
        t0 = np.datetime64(_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None), "us")
        objs = ([RootObject(properties=root_properties(cfg, fs, {"description": "Continuous record"}))] if new else [])
        objs.append(GroupObject(self.group, properties={"segment": segment, "start_time": str(t0)}))
        for c in self.channels:
            objs.append(ChannelObject(self.group, c.name, np.zeros(0), properties=_chan_props(c, fs, 0, t0)))
        self._w.write_segment(objs)

    def write(self, x: np.ndarray, trig: Optional[np.ndarray] = None):
        objs = [ChannelObject(self.group, c.name, np.asarray(row, np.float64)) for c, row in zip(self.channels, x)]
        if self.with_trigger and trig is not None:
            objs.append(ChannelObject(self.group, "Trigger", np.asarray(trig, np.float64)))
        self._w.write_segment(objs)

    def close(self):
        if self._w is not None:
            self._w.__exit__(None, None, None)
            self._w = None


# ----------------------------------------------------------------------------- CSV
def write_hit_log(path: str, hits: Sequence[Hit], channels: Sequence[ChannelConfig]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hit", "segment", "status", "included", "trigger_sample", "trigger_time_s", "wall_time",
                    *[f"peak_{c.name}_{c.units}" for c in channels], "reason", "trigger_pulse_samples"])
        for h in hits:
            w.writerow([h.number, h.segment, h.status, int(h.included), h.sample, f"{h.sample / h.fs:.6f}",
                        h.wall_time, *[f"{p:.6g}" for p in h.peaks], h.reason, " ".join(map(str, h.pulses))])
    return path


def write_trigger_log(path: str, pulses: Sequence[dict]) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pulse", "segment", "reason", "hit", "sample_index", "time_s", "sample_exact",
                    "requested_at_sample", "delay_after_request_ms"])
        for i, p in enumerate(pulses, 1):
            w.writerow([i, p["segment"], p["reason"], p["hit"], p["sample"], f"{p['time_s']:.6f}",
                        int(p["exact"]), p["requested_at_sample"], f"{p['latency_ms']:.2f}"])
    return path


# ----------------------------------------------------------------------------- one measurement point
def point_folder(cfg: AppConfig) -> str:
    s = cfg.save
    tag = f"P{s.point_node}{s.point_dir.replace('+', 'p').replace('-', 'm')}" if s.roving else "fixed"
    base = os.path.join(s.folder, f"{s.basename}_{tag}_{_dt.datetime.now():%Y%m%d_%H%M%S}")
    out, k = base, 2
    while os.path.exists(out):
        out, k = f"{base}_{k}", k + 1
    return out


def save_point(folder: str, cfg: AppConfig, session: PointSession, pulses: Sequence[dict], fs: float,
               log=lambda s: None) -> List[str]:
    """Write everything chosen for one point into `folder`. Returns the files written."""
    os.makedirs(folder, exist_ok=True)
    s = cfg.save
    base = os.path.join(folder, s.basename)
    chans = session.channels
    ref = chans[session.ref_i]
    resp = [chans[i] for i in range(len(chans)) if i != session.ref_i]
    files = []
    res = session.result()                       # None without response channels (time data only)
    n_inc = session.n_included()
    if session.time_only:
        info = f"Impact test, time data only, {n_inc} hits, fs {fs:g} Hz, N {session.n}"
    else:
        info = (f"Impact test, {n_inc} avg, fs {fs:g} Hz, N {session.n}, "
                f"{cfg.acq.resp_window.split(' ')[0]} window")
    meta = {"saved": _dt.datetime.now().isoformat(timespec="seconds"), "fs_actual": fs,
            "n_averages": n_inc, "time_only": session.time_only, "hits": session.summary(),
            "exp_tau_s": session.win.tau if np.isfinite(session.win.tau) else None}
    with open(os.path.join(folder, "setup.json"), "w", encoding="utf-8") as f:
        d = json.loads(cfg.to_json())
        d["run"] = meta
        json.dump(d, f, indent=2)
    files.append(os.path.join(folder, "setup.json"))
    if res is not None and s.save_uff:
        for k, on in (("H1", s.uff_h1), ("H2", s.uff_h2), ("Hv", s.uff_hv), ("Coherence", s.uff_coherence)):
            if on:
                data = res.coherence if k == "Coherence" else getattr(res, k)
                files.append(write_frf_uff(f"{base}_{k}.uff", k, res.freqs[1:], data[:, 1:], resp, ref, info))
        if s.uff_auto:
            files.append(write_auto_uff(f"{base}_AutoSpectra.uff", res.freqs[1:], res.Gxx[1:], res.Gyy[:, 1:],
                                        resp, ref, info))
    inc = [h for h in session.hits if h.included]
    if s.uff_time and inc:
        files.append(write_time_uff(f"{base}_time.uff", inc, chans, ref, info))
    if s.save_hits_tdms:
        hits = session.hits if s.save_rejected else inc
        if hits:
            files.append(write_hits_tdms(f"{base}_hits.tdms", cfg, fs, chans, hits, {"run_meta": json.dumps(meta)}))
    files.append(write_hit_log(f"{base}_hit_log.csv", session.hits, chans))
    if cfg.trigger.enabled and not cfg.trigger.frame_sync:
        files.append(write_trigger_log(f"{base}_trigger_log.csv", pulses))
    for p in files:
        log(f"  saved {os.path.basename(p)}")
    return files
