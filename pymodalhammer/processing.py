# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Impact-hammer signal processing: windows, quality checks of a hit, spectra, H1/H2/Hv/coherence
from the hits that are included, and the measurement-point session that keeps every hit.

Convention: X = reference (hammer force) spectrum, Y = response spectrum.
    Gxx = <|X|²>,  Gyy = <|Y|²>,  Gxy = <X* · Y>
    H1  = Gxy / Gxx                 (noise on the response)
    H2  = Gyy / Gxy*                (noise on the force)
    Hv  = Gxy / (Gxx − λmin)        (total least squares, between H1 and H2)
    γ²  = |Gxy|² / (Gxx · Gyy)
Spectra are Fourier transforms of the windowed transient, X(f) = DFT(x·w) / fs  [EU·s], so a
single hit's spectrum does not depend on the record length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import (ASK, FORCE_WINDOWS, NOCHECK, REJECT, RESP_WINDOWS, AcquisitionConfig,
                     ChannelConfig)

_EPS = 1e-300

# hit status
ACCEPTED, FLAGGED, REJECTED, PENDING, REMOVED, PROBE = "accepted", "flagged", "rejected", "pending", "removed", "probe"
STATUS_COLORS = {ACCEPTED: (39, 174, 96), FLAGGED: (243, 156, 18), REJECTED: (192, 57, 43),
                 PENDING: (52, 152, 219), REMOVED: (127, 140, 141), PROBE: (142, 68, 173)}


# ----------------------------------------------------------------------------- windows
def force_window(acq: AcquisitionConfig, n: int, fs: float, pre: int) -> np.ndarray:
    """1 from the record start to trigger + width, then a half-cosine to 0 over the taper."""
    w = np.ones(n)
    if acq.force_window != FORCE_WINDOWS[0]:
        return w
    a = pre + int(round(acq.force_width_ms * 1e-3 * fs))
    t = max(0, int(round(acq.force_taper_ms * 1e-3 * fs)))
    a = min(max(a, 1), n)
    w[a:] = 0.0
    if t > 0:
        m = np.arange(min(t, n - a))
        w[a:a + m.size] = 0.5 * (1 + np.cos(np.pi * (m + 1) / (t + 1)))
    return w


def exp_tau(acq: AcquisitionConfig, n: int, fs: float, pre: int) -> float:
    """Time constant τ [s] of the exponential window (∞ for rectangular)."""
    if acq.resp_window != RESP_WINDOWS[0]:
        return float("inf")
    T = max(n - 1 - pre, 1) / fs
    return T / max(-np.log(max(acq.exp_end_pct, 1e-6) / 100.0), 1e-9)


def exp_window(acq: AcquisitionConfig, n: int, fs: float, pre: int) -> np.ndarray:
    """1 up to the trigger point, then exp(−t/τ) so that the last sample is end_pct % of 1."""
    tau = exp_tau(acq, n, fs, pre)
    if not np.isfinite(tau):
        return np.ones(n)
    t = np.clip(np.arange(n) - pre, 0, None) / fs
    return np.exp(-t / tau)


def added_damping(tau: float, f: np.ndarray) -> np.ndarray:
    """Damping ratio an exponential window adds to a mode at frequency f: ζ_w = 1 / (2π·f·τ).
    Subtract it from identified damping ratios."""
    if not np.isfinite(tau):
        return np.zeros_like(np.asarray(f, dtype=float))
    return 1.0 / (2 * np.pi * np.asarray(f, dtype=float) * tau)


@dataclass
class Windows:
    force: np.ndarray       # applied to the reference: force window × exponential
    resp: np.ndarray        # applied to every other channel: exponential
    w_force_only: np.ndarray
    w_exp: np.ndarray
    tau: float

    @classmethod
    def make(cls, acq: AcquisitionConfig, n: int, fs: float, pre: int) -> "Windows":
        wf = force_window(acq, n, fs, pre)
        we = exp_window(acq, n, fs, pre)
        return cls(force=wf * we, resp=we, w_force_only=wf, w_exp=we, tau=exp_tau(acq, n, fs, pre))


# ----------------------------------------------------------------------------- checks
@dataclass
class Check:
    name: str               # 'Overload', 'Double hit', 'Force level', 'Decay'
    ok: bool
    message: str
    action: str             # what the setup says to do if it fails


def detect_double_hit(f: np.ndarray, pre: int, fs: float, pct: float):
    """(is_double, ratio %, time of the second peak after the first [ms]). After the main pulse has
    ended (force below 5 % of its peak), any later |force| above pct % of the peak is a second hit."""
    x = np.abs(f)
    i0 = int(np.argmax(x))
    pk = x[i0]
    if pk <= 0:
        return False, 0.0, 0.0
    below = np.flatnonzero(x[i0:] < 0.05 * pk)
    if below.size == 0:
        return False, 0.0, 0.0
    end = i0 + int(below[0]) + max(1, int(0.0003 * fs))      # 0.3 ms past the pulse end
    if end >= x.size:
        return False, 0.0, 0.0
    j = end + int(np.argmax(x[end:]))
    r = 100.0 * x[j] / pk
    return r >= pct, r, (j - i0) / fs * 1000.0


def evaluate_hit(data: np.ndarray, channels: Sequence[ChannelConfig], ref_i: int, acq: AcquisitionConfig,
                 fs: float, pre: int):
    """Quality checks of one hit record. Returns (checks, status, reason) where status is
    'accepted', 'flagged', 'rejected' or 'pending' (= ask me / manual review)."""
    checks: List[Check] = []
    peaks = np.max(np.abs(data), axis=1)
    if acq.overload_action != NOCHECK:
        over = [c.name for c, p in zip(channels, peaks) if p >= c.max_range * acq.overload_pct / 100.0]
        checks.append(Check("Overload", not over,
                            ("overload: " + ", ".join(over)) if over else "no overload", acq.overload_action))
    if ref_i >= 0:
        f = data[ref_i]
        if acq.double_hit_action != NOCHECK:
            dbl, r, dt = detect_double_hit(f, pre, fs, acq.double_hit_pct)
            checks.append(Check("Double hit", not dbl,
                                f"double hit: 2nd peak {r:.0f} % after {dt:.1f} ms" if dbl
                                else f"single hit (largest later peak {r:.1f} %)", acq.double_hit_action))
        if acq.force_action != NOCHECK:
            pk = peaks[ref_i]
            u = channels[ref_i].units
            ok = acq.force_min <= pk <= acq.force_max
            checks.append(Check("Force level", ok,
                                f"force {pk:.4g} {u} " + ("in" if ok else "outside")
                                + f" {acq.force_min:g}…{acq.force_max:g} {u}", acq.force_action))
        if acq.decay_action != NOCHECK and len(channels) > 1:       # no responses: nothing to decay
            n = data.shape[1]
            tail = max(8, n // 20)
            worst, who = 0.0, ""
            for i, (c, p) in enumerate(zip(channels, peaks)):
                if i == ref_i or p <= 0:
                    continue
                r = 100.0 * np.sqrt(2.0) * np.sqrt(np.mean(data[i, -tail:] ** 2)) / p
                if r > worst:
                    worst, who = r, c.name
            ok = worst < acq.decay_pct
            checks.append(Check("Decay", ok, f"{who} still at {worst:.1f} % of its peak at the record end"
                                if not ok else f"responses decayed (≤ {worst:.1f} % at the end)", acq.decay_action))
    failed = [c for c in checks if not c.ok]
    reason = "; ".join(c.message for c in failed)
    if any(c.action == REJECT for c in failed):
        status = REJECTED
    elif any(c.action == ASK for c in failed):
        status = PENDING
    elif failed:
        status = FLAGGED
    else:
        status = ACCEPTED
    if acq.manual:
        status = PENDING
    return checks, status, reason


# ----------------------------------------------------------------------------- hits
@dataclass
class Hit:
    number: int                     # hit number within the point (1, 2, …), in detection order
    segment: int                    # acquisition segment (each Start/Continue starts a new one)
    sample: int                     # trigger sample index from the start of its segment
    fs: float
    pre: int
    data: np.ndarray                # (n_channels, N) in engineering units
    checks: List[Check] = field(default_factory=list)
    status: str = ACCEPTED
    reason: str = ""
    wall_time: str = ""
    pulses: List[int] = field(default_factory=list)        # trigger-output sample indices tied to this hit
    trig_wave: Optional[np.ndarray] = None                  # trigger output during the record (if any)
    spectra: Optional[np.ndarray] = None                    # (n_channels, nf) complex, windowed

    @property
    def included(self) -> bool:
        return self.status in (ACCEPTED, FLAGGED)

    @property
    def peaks(self) -> np.ndarray:
        return np.max(np.abs(self.data), axis=1)

    @property
    def time_s(self) -> float:
        return self.sample / self.fs


def prepare(data: np.ndarray, pre: int, remove_offset: bool) -> np.ndarray:
    """Subtract each channel's pre-trigger mean (sensor offset, AC-coupling drift)."""
    if not remove_offset or pre < 4:
        return data
    return data - np.mean(data[:, :pre], axis=1, keepdims=True)


def hit_spectra(data: np.ndarray, ref_i: int, win: Windows, fs: float) -> np.ndarray:
    """Windowed Fourier transform of every channel [EU·s]: the reference gets force × exponential,
    the others the exponential window."""
    w = np.tile(win.resp, (data.shape[0], 1))
    if ref_i >= 0:
        w[ref_i] = win.force
    return np.fft.rfft(data * w, axis=-1) / fs


def estimates(Gxx, Gyy, Gxy):
    """H1, H2, Hv, coherence from averaged spectra. Gxx (nf,) broadcasts against Gyy/Gxy (n, nf)."""
    Gxx = np.asarray(Gxx)[None, :] if np.ndim(Gyy) > np.ndim(Gxx) else np.asarray(Gxx)
    absGxy2 = np.abs(Gxy) ** 2
    H1 = Gxy / (Gxx + _EPS)
    H2 = Gyy / (np.conj(Gxy) + _EPS)
    lam_min = 0.5 * (Gxx + Gyy) - np.sqrt((0.5 * (Gxx - Gyy)) ** 2 + absGxy2)
    Hv = Gxy / (Gxx - lam_min + _EPS)
    coh = np.clip(absGxy2 / (Gxx * Gyy + _EPS), 0.0, 1.0)
    return H1, H2, Hv, coh


@dataclass
class FRFResult:
    freqs: np.ndarray
    H1: np.ndarray                  # (n_resp, nf)
    H2: np.ndarray
    Hv: np.ndarray
    coherence: np.ndarray
    Gxx: np.ndarray                 # (nf,)     averaged |X|²  [EU²·s²]
    Gyy: np.ndarray                 # (n_resp, nf)
    n_avg: int
    resp_i: List[int]
    ref_i: int


def average(spectra: Sequence[np.ndarray], ref_i: int, freqs: np.ndarray) -> Optional[FRFResult]:
    """FRFs from the spectra (n_ch, nf) of the included hits (None without hits or responses)."""
    if not spectra or ref_i < 0 or np.shape(spectra[0])[0] < 2:
        return None
    S = np.stack(spectra)                       # (k, n_ch, nf)
    n_ch = S.shape[1]
    resp = [i for i in range(n_ch) if i != ref_i]
    X = S[:, ref_i, :]
    Y = S[:, resp, :]
    Gxx = np.mean(np.abs(X) ** 2, axis=0)
    Gyy = np.mean(np.abs(Y) ** 2, axis=0)
    Gxy = np.mean(np.conj(X)[:, None, :] * Y, axis=0)
    H1, H2, Hv, coh = estimates(Gxx, Gyy, Gxy)
    return FRFResult(freqs, H1, H2, Hv, coh, Gxx, Gyy, S.shape[0], resp, ref_i)


def force_rolloff(freqs: np.ndarray, Xf: np.ndarray, db: float = -10.0) -> float:
    """First frequency above which the force spectrum stays below `db` re its low-frequency level.
    The hammer tip excites the structure reliably up to about the −10 dB (to −20 dB) frequency."""
    a = np.abs(Xf)
    if a.size < 4:
        return float("nan")
    k = max(1, int(0.02 * a.size))
    ref = np.max(a[1:k + 1]) if k >= 1 else a[1]
    if ref <= 0:
        return float("nan")
    lev = 20 * np.log10(a / ref + _EPS)
    above = np.flatnonzero(lev >= db)
    if above.size == 0:
        return float(freqs[1])
    last = int(above[-1])
    return float(freqs[min(last + 1, freqs.size - 1)])


# ----------------------------------------------------------------------------- point session
class PointSession:
    """All hits of one measurement point and the running average of the included ones. Lives in the
    GUI thread: the acquisition thread only detects and checks hits."""

    def __init__(self, channels: List[ChannelConfig], ref_i: int, acq: AcquisitionConfig, fs: float):
        self.channels = channels
        self.ref_i = ref_i
        self.acq = acq
        self.fs = fs
        self.n = acq.block_size
        self.pre = acq.pre_samples
        self.win = Windows.make(acq, self.n, fs, self.pre)
        self.freqs = np.arange(self.n // 2 + 1) * fs / self.n
        self.hits: List[Hit] = []
        self._result: Optional[FRFResult] = None
        self._dirty = True
        self.segment = 0

    @property
    def time_only(self) -> bool:
        """Only the hammer channel: hits are kept as time data, there is no FRF."""
        return len(self.channels) < 2

    def set_fs(self, fs: float):
        """The sample rate DAQmx actually uses (delta-sigma modules coerce it)."""
        if abs(fs - self.fs) > 1e-9 * fs and not self.hits:
            self.fs = fs
            self.win = Windows.make(self.acq, self.n, fs, self.pre)
            self.freqs = np.arange(self.n // 2 + 1) * fs / self.n

    def add(self, hit: Hit) -> Hit:
        hit.spectra = hit_spectra(hit.data, self.ref_i, self.win, self.fs)
        self.hits.append(hit)
        self._dirty = True
        return hit

    def set_status(self, i: int, status: str, reason: str = None):
        self.hits[i].status = status
        if reason is not None:
            self.hits[i].reason = reason
        self._dirty = True

    def delete(self, indices) -> List["Hit"]:
        """Remove hits completely: out of the list, the average and the files saved later."""
        gone = [self.hits[i] for i in sorted(set(indices))]
        self.hits = [h for h in self.hits if all(h is not g for g in gone)]
        self._dirty = True
        return gone

    def n_included(self) -> int:
        return sum(h.included for h in self.hits)

    def pending(self) -> List[int]:
        return [i for i, h in enumerate(self.hits) if h.status == PENDING]

    def result(self) -> Optional[FRFResult]:
        if self._dirty:
            self._result = average([h.spectra for h in self.hits if h.included], self.ref_i, self.freqs)
            self._dirty = False
        return self._result

    def complete(self) -> bool:
        return self.n_included() >= self.acq.n_averages

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for h in self.hits:
            out[h.status] = out.get(h.status, 0) + 1
        return out
