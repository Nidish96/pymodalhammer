# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Acquisition engine (runs in a worker thread).

The input task runs continuously. Every read (`chunk` samples) is appended to a ring buffer and
searched for a trigger crossing on the trigger channel. A crossing at absolute sample s starts a
record [s − pre, s − pre + N). When its last sample has arrived the record is checked
(overload, double hit, force level, decay) and handed to the GUI, which keeps the point's
average. While a hit waits for your decision (Ask me / manual review) the trigger stays
disarmed; `rearm()` arms it again.

External trigger output
    Hardware-timed: a continuous output task on the AI sample clock. One output chunk is written
    per read, `queue` chunks ahead. A pulse requested now goes into the next chunk written, so it
    starts at a known absolute sample index (logged), about queue × chunk after the request.
    On-demand: a static line is set active, then idle after the pulse width (software timing).
    Frame sync: the AI sample clock itself is routed to a PFI pin (one pulse per sample).
"""

from __future__ import annotations

import datetime as _dt
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .backends import trigger_levels
from .config import TRIGGER_SLOPES, AppConfig
from .processing import PENDING, PROBE, Hit, evaluate_hit, prepare

M_START, M_DETECT, M_ACCEPT, M_FIRST, M_DONE, M_FRAME = range(6)


@dataclass
class Callbacks:
    log: Callable[[str], None] = lambda s: None
    scope: Callable[..., None] = lambda *a: None     # (data, start, fs, peaks, backlog, trigger output or None)
    hit: Callable[[Hit], None] = lambda h: None
    pulse: Callable[[dict], None] = lambda d: None
    started: Callable[[float], None] = lambda fs: None


@dataclass
class EngineSummary:
    fs: float = 0.0
    samples: int = 0
    hits: int = 0
    aborted: bool = False
    stream_file: str = ""
    error: str = ""
    pulses: List[dict] = field(default_factory=list)


class HammerEngine:
    def __init__(self, cfg: AppConfig, backend, cb: Callbacks, probe: bool = False, segment: int = 1,
                 first_hit_number: int = 1, stream_path: str = "", scope_s: float = 2.0):
        self.cfg = cfg
        self.backend = backend
        self.cb = cb
        self.probe = probe
        self.segment = segment
        self.channels = cfg.labelled_channels()
        self.ref_i = cfg.reference_index()
        self.trig_i = cfg.trigger_index()
        self.acq = cfg.acq
        self.N = self.acq.block_size
        self.pre = self.acq.pre_samples
        t = cfg.trigger
        self.trig = t if (t.enabled and (t.in_probe or not probe)) else None
        self.tmode = self.trig.mode_index() if self.trig else -1
        self.stream_path = stream_path
        self.scope_s = scope_s
        self._stop = threading.Event()
        self._armed = threading.Event()
        self._armed.set()
        self._pulse_req: "queue.Queue[tuple]" = queue.Queue()
        self._next_hit = first_hit_number
        self._first_hit_pending = True
        self.summary = EngineSummary()
        self.fs = self.acq.fs

    # ------------------------------------------------------------------ thread-safe controls
    def request_stop(self):
        self._stop.set()

    def rearm(self):
        self._armed.set()

    def disarm(self):
        self._armed.clear()

    def new_point(self, node: int = None):
        """Roving hammer: the next point starts without restarting the acquisition. Hit numbers
        start again at 1; the simulator moves its hammer to `node`."""
        self._first_hit_pending = True
        self._next_hit = 1
        if node is not None and hasattr(self.backend, "set_point"):
            self.backend.set_point(node)
        self._armed.set()

    def request_pulse(self, reason: str, hit: int = 0):
        """Send one pulse now (hardware-timed: in the next output chunk)."""
        if self.trig is not None and not self.trig.frame_sync:
            self._pulse_req.put((reason, hit, time.perf_counter()))

    def wants(self, mode: int) -> bool:
        return self.trig is not None and self.tmode == mode

    # ------------------------------------------------------------------ main loop
    def run(self) -> EngineSummary:
        a, be, cb = self.acq, self.backend, self.cb
        n_ch = len(self.channels)
        try:
            fs, notes = be.configure(a.chunk_samples())
        except Exception:
            be.close()
            raise
        self.fs = fs
        self.summary.fs = fs
        C = a.chunk_samples(fs) if abs(fs - a.fs) > 1e-6 * a.fs else a.chunk_samples()
        if getattr(be, "n", C) != C:
            C = be.n
        self.C = C
        for s in notes:
            cb.log(s)
        cb.log(f"{'PROBE MODE: ' if self.probe else ''}fs = {fs:.6g} Hz, record {self.N} samples = "
               f"{self.N / fs:.3f} s (Δf = {fs / self.N:.4g} Hz), pre-trigger {self.pre} samples, "
               f"read size {C} samples ({C / fs * 1000:.0f} ms)")
        # ring buffer large enough for a full record plus a few reads
        cap = self.N + 4 * C + self.pre
        buf = np.zeros((n_ch, cap))
        tbuf = np.zeros(cap)                       # trigger output written, aligned with the inputs
        buf_start = 0                              # absolute index of buf[:, 0]
        filled = 0
        scope_n = int(max(C, self.scope_s * fs))
        # output pulse bookkeeping (hardware-timed)
        on_v, idle_v = trigger_levels(self.trig) if self.trig is not None and not self.trig.frame_sync else (1.0, 0.0)
        hw = bool(getattr(be, "hw_trigger", False))
        q = a.out_chunks(fs)
        out_written = 0
        out_hist = []                              # chunks written but not yet "acquired" (for tbuf alignment)
        scheduled: List[tuple] = []                # hardware-timed pulses (start sample, width) not yet written
        pulses: List[dict] = []
        settle_until = int(a.settle_s * fs)
        rearm_at = settle_until
        rearmed = False
        capture: Optional[int] = None             # absolute trigger index of the record being collected
        capture_pulses: List[int] = []
        lvl = a.trig_level
        low = lvl * a.rearm_pct / 100.0
        slope = TRIGGER_SLOPES.index(a.trig_slope) if a.trig_slope in TRIGGER_SLOPES else 0
        stream = None
        if self.stream_path:
            from .export import TdmsStreamWriter
            stream = TdmsStreamWriter(self.stream_path, self.cfg, fs, self.channels, self.segment,
                                      with_trigger=hw)
            self.summary.stream_file = self.stream_path
        last_scope = 0.0

        def queue_pulse(reason, hit_no, t_req=None):
            if self.trig is None or self.trig.frame_sync:
                return None
            w = self.trig.pulse_samples
            rec = {"segment": self.segment, "reason": reason, "hit": hit_no,
                   "requested_at_sample": int(total)}
            if hw:
                # next chunk to be written; after the previous pulse plus an idle gap of one width
                start = out_written
                if scheduled:
                    start = max(start, scheduled[-1][0] + 2 * w)
                scheduled.append((start, w))
                rec["sample"] = int(start)
                rec["exact"] = True
            else:
                be.pulse_now(True)
                rec["sample"] = int(be.samples_acquired()) if be.samples_acquired() >= 0 else int(total)
                rec["exact"] = False
                threading.Timer(max(self.trig.pulse_samples / fs, 0.001), lambda: be.pulse_now(False)).start()
            rec["time_s"] = rec["sample"] / fs
            rec["latency_ms"] = (rec["sample"] - rec["requested_at_sample"]) / fs * 1000.0
            rec["pulse"] = len(pulses) + 1
            pulses.append(rec)
            cb.pulse(rec)
            return rec

        def next_out_chunk():
            nonlocal out_written
            t = np.full(C, idle_v)
            a0, a1 = out_written, out_written + C
            for st, w in scheduled:
                lo, hi = max(st, a0), min(st + w, a1)
                if hi > lo:
                    t[lo - a0:hi - a0] = on_v
            while scheduled and scheduled[0][0] + scheduled[0][1] <= a1:
                scheduled.pop(0)
            out_written += C
            return t

        total = 0
        try:
            if hw:
                if self.wants(M_START):
                    queue_pulse("acquisition start", 0)
                for _ in range(q):
                    t = next_out_chunk()
                    be.write_trigger(t)
                    out_hist.append(t)
            be.start()
            if not hw and self.wants(M_START):
                queue_pulse("acquisition start", 0)
            cb.started(fs)
            while not self._stop.is_set():
                x = be.read_chunk()
                s0 = total
                total += C
                tr_chunk = out_hist.pop(0) if hw else None
                # ---- pulses requested by the GUI (accepted hit, point complete, manual)
                while True:
                    try:
                        reason, hit_no, _ = self._pulse_req.get_nowait()
                    except queue.Empty:
                        break
                    rec = queue_pulse(reason, hit_no)
                    if rec is not None and capture is not None and hit_no == self._next_hit:
                        capture_pulses.append(rec["sample"])
                # ---- ring buffer
                if filled + C > cap:
                    drop = filled + C - cap
                    buf[:, :filled - drop] = buf[:, drop:filled]
                    tbuf[:filled - drop] = tbuf[drop:filled]
                    buf_start += drop
                    filled -= drop
                buf[:, filled:filled + C] = x
                tbuf[filled:filled + C] = tr_chunk if tr_chunk is not None else 0.0
                filled += C
                if stream is not None:
                    stream.write(x, tr_chunk)
                # ---- trigger search
                if capture is None and self._armed.is_set() and total > rearm_at:
                    i_from = max(s0, rearm_at, self.pre)
                    seg = buf[self.trig_i, i_from - buf_start: total - buf_start] if self.trig_i >= 0 else np.zeros(0)
                    v = seg if slope == 0 else (-seg if slope == 1 else np.abs(seg))
                    j = None
                    if v.size:
                        if not rearmed:
                            below = np.flatnonzero(v < low)
                            if below.size:
                                rearmed = True
                                v2 = v[below[0]:]
                                k = np.flatnonzero(v2 >= lvl)
                                j = int(below[0] + k[0]) if k.size else None
                        else:
                            k = np.flatnonzero(v >= lvl)
                            j = int(k[0]) if k.size else None
                    if j is not None:
                        capture = i_from + j
                        capture_pulses = []
                        if self.wants(M_DETECT):
                            rec = queue_pulse("hit detected", self._next_hit)
                            capture_pulses.append(rec["sample"])
                        if self.wants(M_FIRST) and self._first_hit_pending:
                            rec = queue_pulse("first hit of the point", self._next_hit)
                            capture_pulses.append(rec["sample"])
                        self._first_hit_pending = False
                # ---- record complete?
                if capture is not None and capture - self.pre + self.N <= total:
                    r0 = capture - self.pre - buf_start
                    data = prepare(buf[:, r0:r0 + self.N].copy(), self.pre, a.remove_offset)
                    twave = tbuf[r0:r0 + self.N].copy() if hw else None
                    hit = Hit(number=self._next_hit, segment=self.segment, sample=int(capture), fs=fs,
                              pre=self.pre, data=data, trig_wave=twave, pulses=list(capture_pulses),
                              wall_time=_dt.datetime.now().strftime("%H:%M:%S.%f")[:-3])
                    hit.checks, hit.status, hit.reason = evaluate_hit(
                        data, self.channels, self.ref_i, a, fs, self.pre)
                    if self.probe:
                        hit.status = PROBE                  # checks are shown as advice only
                    else:
                        if hit.status == PENDING:
                            self._armed.clear()            # wait for the decision
                        elif hit.included and self.wants(M_ACCEPT):
                            rec = queue_pulse("accepted hit", hit.number)
                            hit.pulses.append(rec["sample"])
                    self._next_hit += 1
                    self.summary.hits += 1
                    cb.hit(hit)
                    rearm_at = capture - self.pre + self.N + int(a.holdoff_s * fs)
                    rearmed = False
                    capture = None
                # ---- hardware-timed output: keep the queue full
                if hw:
                    t = next_out_chunk()
                    try:
                        be.write_trigger(t)
                    except Exception as e:
                        if getattr(e, "error_code", None) == -200621 or "underflow" in str(e).lower():
                            raise RuntimeError(
                                f"The external-trigger output ran dry (DAQmx output underflow): the program was held up "
                                f"for more than the output queue ({a.out_queue_ms:g} ms). Increase 'Output queue' on "
                                "the Acquisition Setup tab (1000–2000 ms is safe on a busy PC; the pulse delay grows "
                                "with it), close other programs, or use On-demand timing with a DO line.") from e
                        raise
                    out_hist.append(t)
                # ---- scope (rate-limited)
                now = time.perf_counter()
                if now - last_scope >= 0.08:
                    last_scope = now
                    m = min(filled, scope_n)
                    peaks = np.max(np.abs(x), axis=1)
                    cb.scope(buf[:, filled - m:filled].copy(), total - m, fs, peaks,
                             int(be.backlog()), tbuf[filled - m:filled].copy() if hw else None)
        except Exception:
            self.summary.aborted = True
            raise
        finally:
            self.summary.samples = total
            self.summary.pulses = pulses
            self.summary.aborted = self.summary.aborted or self._stop.is_set()
            try:
                be.stop()
            finally:
                be.close()
                if stream is not None:
                    stream.close()
            if getattr(be, "trigger_edges", None) and hw:
                cb.log(f"(simulation) trigger edges at samples {be.trigger_edges[:10]}"
                       + (" …" if len(be.trigger_edges) > 10 else ""))
        return self.summary
