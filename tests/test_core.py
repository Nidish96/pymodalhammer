# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Headless checks of the acquisition engine, processing and file export (simulation backend).
Run from the project folder:  python -m pytest -q tests   (or: python tests/test_core.py)"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymodalhammer.backends import SimBackend                                     # noqa: E402
from pymodalhammer.config import (FLAG, REJECT, TRIGGER_MODES, TRIGGER_TIMING, AppConfig,  # noqa: E402
                                 default_config)
from pymodalhammer.engine import Callbacks, HammerEngine                          # noqa: E402
from pymodalhammer.export import point_folder, save_point                         # noqa: E402
from pymodalhammer.processing import (ACCEPTED, PointSession, detect_double_hit,   # noqa: E402
                                     force_rolloff)
from pymodalhammer.results import frf_table, load_hits, recompute, scan_run       # noqa: E402


def run_sim(cfg: AppConfig, seconds: float, probe=False, seed=3, **simkw):
    hits, pulses, logs = [], [], []
    be = SimBackend(cfg, probe=probe, realtime=False, seed=seed, **simkw)
    eng = HammerEngine(cfg, be, Callbacks(log=logs.append, hit=hits.append, pulse=pulses.append), probe=probe)
    n_reads = int(seconds / (cfg.acq.chunk_ms / 1000))
    orig = be.read_chunk
    count = [0]

    def limited():
        count[0] += 1
        if count[0] >= n_reads:
            eng.request_stop()
        return orig()

    be.read_chunk = limited
    summ = eng.run()
    return hits, pulses, logs, summ, be


def test_detects_hits_and_frf():
    cfg = default_config()
    hits, pulses, logs, summ, be = run_sim(cfg, 30)
    assert len(hits) >= 6, logs
    s = PointSession(cfg.labelled_channels(), cfg.reference_index(), cfg.acq, summ.fs)
    for h in hits:
        s.add(h)
    res = s.result()
    assert res is not None and res.n_avg >= 4
    # the H1 peak near the first simulated mode
    f = res.freqs
    j = np.argmax(np.abs(res.H1[0][(f > 50) & (f < 200)]))
    fpk = f[(f > 50) & (f < 200)][j]
    assert abs(fpk - be.fn[0]) < 3 * f[1], (fpk, be.fn[0])
    # coherence high around the mode
    assert np.median(res.coherence[0][(f > 80) & (f < 800)]) > 0.9
    # pre-trigger is respected: the trigger point is at sample `pre`
    h = hits[0]
    assert h.data[0, h.pre] >= cfg.acq.trig_level * 0.99
    assert np.all(np.abs(h.data[0, :h.pre - 2]) < cfg.acq.trig_level)


def test_double_hit_detector():
    fs, n = 10240, 4096
    t = np.arange(n) / fs
    f = np.where((t > 0.01) & (t < 0.011), 200 * np.sin(np.pi * (t - 0.01) / 0.001), 0.0)
    assert not detect_double_hit(f, 50, fs, 10)[0]
    f2 = f + np.where((t > 0.016) & (t < 0.017), 60 * np.sin(np.pi * (t - 0.016) / 0.001), 0.0)
    dbl, r, dt = detect_double_hit(f2, 50, fs, 10)
    assert dbl and 25 < r < 35 and 5 < dt < 7


def test_rejections_and_removal():
    cfg = default_config()
    cfg.acq.double_hit_action = REJECT
    cfg.acq.overload_action = REJECT
    hits, *_ , summ, be = run_sim(cfg, 80, seed=11)
    st = {h.status for h in hits}
    assert ACCEPTED in st
    s = PointSession(cfg.labelled_channels(), cfg.reference_index(), cfg.acq, summ.fs)
    for h in hits:
        s.add(h)
    n0 = s.n_included()
    first = next(i for i, h in enumerate(s.hits) if h.included)
    n_hits = len(s.hits)
    s.delete([first])
    assert len(s.hits) == n_hits - 1 and s.n_included() == n0 - 1
    assert s.result().n_avg == n0 - 1


def test_trigger_output_hardware_timed():
    cfg = default_config()
    cfg.trigger.enabled = True
    cfg.trigger.mode = TRIGGER_MODES[1]                 # every detected hit
    cfg.trigger.timing = TRIGGER_TIMING[0]
    cfg.trigger.pulse_samples = 20
    hits, pulses, logs, summ, be = run_sim(cfg, 20)
    assert pulses and len(pulses) == len(hits) + (1 if summ.samples and pulses[-1]["hit"] > len(hits) else 0) \
        or len(pulses) >= len(hits)
    # every logged pulse appears as an edge on the simulated output line, at the logged sample
    edges = set(be.trigger_edges)
    for p in pulses:
        if p["sample"] < summ.samples:
            assert p["sample"] in edges, (p, sorted(edges)[:10])
    # the pulse follows the hit by ≤ (queue + 2) reads
    C = cfg.acq.chunk_samples()
    for h in hits:
        assert h.pulses and 0 < h.pulses[0] - h.sample <= (cfg.acq.out_chunks() + 2) * C
        assert h.trig_wave is not None


def test_start_pulse_and_accept_mode():
    cfg = default_config()
    cfg.trigger.enabled = True
    cfg.trigger.mode = TRIGGER_MODES[0]
    hits, pulses, logs, summ, be = run_sim(cfg, 5)
    assert len(pulses) == 1 and pulses[0]["sample"] == 0 and 0 in be.trigger_edges
    cfg.trigger.mode = TRIGGER_MODES[2]
    cfg.acq.decay_action = FLAG
    hits, pulses, logs, summ, be = run_sim(cfg, 25)
    assert len(pulses) == sum(h.included for h in hits)


def test_save_and_reload():
    cfg = default_config()
    cfg.trigger.enabled = True
    with tempfile.TemporaryDirectory() as d:
        cfg.save.folder = d
        cfg.save.uff_h2 = cfg.save.uff_hv = cfg.save.uff_auto = cfg.save.uff_time = True
        hits, pulses, logs, summ, be = run_sim(cfg, 25)
        s = PointSession(cfg.labelled_channels(), cfg.reference_index(), cfg.acq, summ.fs)
        for h in hits:
            s.add(h)
        s.delete([0])
        folder = point_folder(cfg)
        files = save_point(folder, cfg, s, pulses, summ.fs)
        names = {os.path.basename(f) for f in files}
        for k in ("H1", "H2", "Hv", "Coherence", "AutoSpectra", "time"):
            assert f"impact_test_{k}.uff" in names, names
        assert "impact_test_hits.tdms" in names and "impact_test_trigger_log.csv" in names
        run = scan_run(folder)
        assert run.ok, run.problems
        hl = load_hits(run)
        assert len(hl) == len(hits) - 1 and hl[0]["number"] == 2
        tab = frf_table(run, prefer_tdms=True)
        tab2 = frf_table(run, prefer_tdms=False)
        for name in ("Acc_1", "Acc_2"):
            a, b = tab[name], tab2[name]
            m = (a["freqs"] > 20) & (a["freqs"] < 1000)
            np.testing.assert_allclose(np.abs(a["H1"][m]), np.abs(np.interp(a["freqs"][m], b["freqs"], np.abs(b["H1"]))),
                                       rtol=1e-6)
        r = recompute(run, [h["group"] for h in hl if h["included"]])
        assert r.n_avg == sum(h.included for h in s.hits)
        assert abs(r.freqs[1] - summ.fs / cfg.acq.block_size) < 1e-9
        assert np.isfinite(force_rolloff(r.freqs, np.sqrt(r.Gxx)))


def test_time_only_hammer_channel():
    """Only the hammer enabled: hits are detected and saved as time data, no FRF files, external
    trigger pulses still sent and logged."""
    cfg = default_config()
    for c in cfg.channels[1:]:
        c.enabled = False
    cfg.trigger.enabled = True
    cfg.trigger.mode = TRIGGER_MODES[1]                 # every detected hit
    assert cfg.time_only
    with tempfile.TemporaryDirectory() as d:
        cfg.save.folder = d
        cfg.save.uff_time = True
        assert cfg.validate() == [], cfg.validate()
        hits, pulses, logs, summ, be = run_sim(cfg, 25)
        assert len(hits) >= 4 and pulses, logs
        assert all(h.data.shape[0] == 1 for h in hits)
        assert all(c.name != "Decay" for c in hits[0].checks)
        s = PointSession(cfg.labelled_channels(), cfg.reference_index(), cfg.acq, summ.fs)
        for h in hits:
            s.add(h)
        assert s.time_only and s.result() is None and s.n_included() > 0
        files = save_point(point_folder(cfg), cfg, s, pulses, summ.fs)
        names = {os.path.basename(f) for f in files}
        assert names == {"setup.json", "impact_test_time.uff", "impact_test_hits.tdms",
                         "impact_test_hit_log.csv", "impact_test_trigger_log.csv"}, names
        run = scan_run(os.path.dirname(files[0]))
        assert run.ok and run.meta["time_only"] and run.meta["n_averages"] == s.n_included()
        assert len(load_hits(run)) == len(s.hits) and frf_table(run) == {}
        assert recompute(run, [h["group"] for h in load_hits(run)]) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
