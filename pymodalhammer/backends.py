# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""DAQ backends with one streaming interface:

    fs_actual, notes = backend.configure(chunk)    # chunk = samples per read
    backend.write_trigger(t)     # hardware-timed trigger output: queue `chunk` samples (call `queue
                                 # depth` times before start, then once per read)
    backend.start()
    data = backend.read_chunk()  # (n_channels, chunk) in engineering units, blocks until available
    backend.pulse_now(active)    # on-demand trigger output: set the line active / idle at once
    backend.stop(); backend.close()

NIDAQBackend - continuous AI; the external trigger is a second task (AO or DO) on the AI clock,
               or the AI sample clock exported to a PFI pin (frame sync), or a static line.
SimBackend   - a 3-mode structure struck by a simulated hammer (random or on request), paced in
               real time; double hits and overloads happen now and then.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, Tuple

import numpy as np

from .config import AppConfig, ChannelConfig, voltage_scaled


# ----------------------------------------------------------------------------- discovery
def discover_devices() -> Dict:
    """List NI devices / physical channels. Never raises."""
    out = {"ok": False, "message": "", "devices": [], "ai": [], "ao": [], "do": [], "pfi": []}
    try:
        import nidaqmx.system
        system = nidaqmx.system.System.local()
        drv = system.driver_version
        for dev in system.devices:
            try:
                out["ai"] += list(dev.ai_physical_chans.channel_names)
                out["ao"] += list(dev.ao_physical_chans.channel_names)
            except Exception:
                pass
            try:
                out["do"] += list(dev.do_lines.channel_names)
            except Exception:
                pass
            try:
                out["pfi"] += [t for t in dev.terminals if "PFI" in t.upper()]
            except Exception:
                pass
            out["devices"].append(f"{dev.name} ({dev.product_type})")
        out["ok"] = True
        out["message"] = (f"NI-DAQmx {drv.major_version}.{drv.minor_version}: "
                          + (", ".join(out["devices"]) if out["devices"] else "no devices found"))
    except Exception as e:
        out["message"] = f"NI-DAQmx not available ({type(e).__name__}: {e})"
    return out


def chassis_of(physical_channel: str) -> str:
    """'cDAQ1Mod1/ai0' -> 'cDAQ1' (asks DAQmx first, falls back to the name)."""
    module = physical_channel.strip().lstrip("/").split("/")[0]
    try:
        import nidaqmx.system
        return nidaqmx.system.Device(module).compact_daq_chassis_device.name
    except Exception:
        m = re.match(r"(.+?)Mod\d+$", module)
        return m.group(1) if m else module


def trigger_levels(trig) -> Tuple[float, float]:
    """(active, idle) output values of the trigger line: volts for AO, 1/0 for DO."""
    hi = trig.level_v if trig.kind == "AO" else 1.0
    low_active = trig.polarity.startswith("Active low")
    return (0.0, hi) if low_active else (hi, 0.0)


def ao_trigger_range(physical: str, level: float) -> float:
    """± range for an AO trigger channel: covers the level, never more than the module allows
    (e.g. ±4.24 V on an NI 9260). Raises ValueError with a clear message if the level does not fit."""
    need = max(abs(level), 1.0)
    try:
        import nidaqmx.system
        dev = nidaqmx.system.Device(physical.strip().lstrip("/").split("/")[0])
        rngs = list(dev.ao_voltage_rngs)
        vmax = max(abs(v) for v in rngs) if rngs else 0.0
        product = dev.product_type
    except Exception:
        return need
    if vmax and abs(level) > vmax + 1e-9:
        raise ValueError(f"Trigger level {level:g} V exceeds the ±{vmax:g} V output range of {product}. "
                         "Lower it on the Acquisition Setup tab (3.3 V is enough for TTL inputs).")
    return min(need, vmax) if vmax else need


def _add_trigger_chan(task, trig, name="Trigger"):
    import nidaqmx.constants as K
    if trig.kind == "AO":
        rng = ao_trigger_range(trig.physical, trig.level_v)
        task.ao_channels.add_ao_voltage_chan(trig.physical, name, min_val=-rng, max_val=rng)
    else:
        task.do_channels.add_do_chan(trig.physical, name, line_grouping=K.LineGrouping.CHAN_PER_LINE)


def _needs_clock(err) -> bool:
    """DAQmx refused a static (on-demand) write: the module only supports sample-clocked output
    (e.g. the delta-sigma NI 9260, error -201087)."""
    return getattr(err, "error_code", None) == -201087 or "specify the Sample Clock rate" in str(err)


def write_trigger_samples(trig, samples, fs: float) -> str:
    """Put `samples` (volts for AO, 0/1 for DO) on the trigger line now, outside a run.
    A short finite, sample-clocked generation is tried first (exact pulse width; the only way on
    modules such as the NI 9260); a static write is the fallback for static-only lines.
    Returns how it was done."""
    import nidaqmx
    import nidaqmx.constants as K
    x = np.asarray(samples, dtype=float)
    try:
        with nidaqmx.Task() as t:
            _add_trigger_chan(t, trig)
            t.timing.cfg_samp_clk_timing(fs, sample_mode=K.AcquisitionType.FINITE, samps_per_chan=max(2, x.size))
            if trig.kind == "AO":
                t.write(x.tolist(), auto_start=False)
            else:
                t.write((x > 0.5).tolist(), auto_start=False)
            t.start()
            t.wait_until_done(timeout=max(2.0, 2 * x.size / fs))
            rate = float(t.timing.samp_clk_rate)
        return f"sample-clocked at {rate:.6g} Hz"
    except ValueError:
        raise
    except Exception as e_clocked:
        try:
            with nidaqmx.Task() as t:
                _add_trigger_chan(t, trig)
                prev = None
                for v in x:
                    if v != prev:
                        t.write(bool(v > 0.5) if trig.kind == "DO" else float(v), auto_start=True)
                        prev = v
                    time.sleep(1.0 / fs)
            return "static (software-timed) writes"
        except Exception:
            raise e_clocked


def send_test_pulse(trig, fs: float) -> str:
    on, off = trigger_levels(trig)
    w = max(1, int(trig.pulse_samples))
    pad = max(4, w)
    how = write_trigger_samples(trig, [off] * pad + [on] * w + [off] * pad, fs)
    return f"Test pulse of {w} samples ({w / fs * 1e6:,.0f} µs) on {trig.physical}, {how}."


# ----------------------------------------------------------------------------- NI hardware
class NIDAQBackend:
    def __init__(self, cfg: AppConfig, probe: bool = False):
        self.cfg = cfg
        self.channels: List[ChannelConfig] = cfg.enabled_channels()
        t = cfg.trigger
        self.trig = t if (t.enabled and (t.in_probe or not probe)) else None
        self.ai = self.out = self.static = None
        self.hw_trigger = False          # True: write_trigger() must be called once per chunk
        self._scale = np.ones(len(self.channels))

    def _add_channel(self, c: ChannelConfig):
        import nidaqmx.constants as K
        ai = self.ai.ai_channels
        r = abs(c.max_range)
        excit_src = K.ExcitationSource.INTERNAL if c.iepe_ma > 0 else K.ExcitationSource.NONE
        excit_a = c.iepe_ma / 1000.0
        t = c.sensor_type
        if t.startswith("Accelerometer"):
            units = K.AccelUnits.G if c.units == "g" else K.AccelUnits.METERS_PER_SECOND_SQUARED
            ch = ai.add_ai_accel_chan(c.physical, c.name, min_val=-r, max_val=r, units=units,
                                      sensitivity=c.sensitivity,
                                      sensitivity_units=K.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                                      current_excit_source=excit_src, current_excit_val=excit_a)
        elif t.startswith("Force"):
            lbf = c.units == "lbf"
            ch = ai.add_ai_force_iepe_chan(
                c.physical, c.name, min_val=-r, max_val=r,
                units=K.ForceUnits.POUNDS if lbf else K.ForceUnits.NEWTONS,
                sensitivity=c.sensitivity,
                sensitivity_units=(K.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_POUND if lbf
                                   else K.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON),
                current_excit_source=excit_src, current_excit_val=excit_a)
        elif t.startswith("Microphone"):
            spl = 20 * np.log10(max(r, 1e-3) / 20e-6)
            ch = ai.add_ai_microphone_chan(c.physical, c.name, mic_sensitivity=c.sensitivity,
                                           max_snd_press_level=float(spl),
                                           current_excit_source=excit_src, current_excit_val=excit_a)
        elif t.startswith("Strain"):
            bridge = getattr(K.StrainGageBridgeType, c.bridge.upper().replace(" ", "_"))
            ch = ai.add_ai_strain_gage_chan(c.physical, c.name, min_val=-r, max_val=r,
                                            strain_config=bridge, voltage_excit_val=c.excit_v,
                                            gage_factor=c.gage_factor, nominal_gage_resistance=c.nominal_r,
                                            poisson_ratio=c.poisson)
        else:
            vmax = r * c.sensitivity / 1000.0
            ch = ai.add_ai_voltage_chan(c.physical, c.name, min_val=-vmax, max_val=vmax)
            if c.iepe_ma > 0:
                ch.ai_excit_src = K.ExcitationSource.INTERNAL
                ch.ai_excit_val = excit_a
        if c.coupling in ("AC", "DC"):
            ch.ai_coupling = K.Coupling.AC if c.coupling == "AC" else K.Coupling.DC
        return 1000.0 / c.sensitivity if voltage_scaled(t) else 1.0

    def configure(self, chunk: int) -> Tuple[float, List[str]]:
        import nidaqmx
        import nidaqmx.constants as K
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        a = self.cfg.acq
        self.n = int(chunk)
        notes = []
        self._set_trigger_idle()
        self.ai = nidaqmx.Task("HammerAI")
        self._scale = np.array([self._add_channel(c) for c in self.channels])
        buf = int(max(10 * a.fs, 20 * self.n))
        self.ai.timing.cfg_samp_clk_timing(a.fs, sample_mode=K.AcquisitionType.CONTINUOUS, samps_per_chan=buf)
        fs = float(self.ai.timing.samp_clk_rate)
        if abs(fs - a.fs) > 1e-6 * a.fs:
            notes.append(f"AI sample rate coerced by hardware: {a.fs:g} → {fs:.6g} Hz")
        chassis = chassis_of(self.channels[0].physical)
        tr = self.trig
        if tr is not None and tr.frame_sync:
            ex = self.ai.export_signals
            ex.export_signal(K.Signal.SAMPLE_CLOCK, tr.physical)
            try:
                ex.samp_clk_output_behavior = K.ExportAction.PULSE
                ex.samp_clk_pulse_polarity = (K.Polarity.ACTIVE_LOW if tr.polarity.startswith("Active low")
                                              else K.Polarity.ACTIVE_HIGH)
            except Exception as e:
                notes.append(f"Note: could not set exported clock polarity ({e}); using the default.")
            notes.append(f"Frame sync: AI sample clock exported to {tr.physical} ({fs:.6g} pulses/s, "
                         "first pulse = sample 0). Set the camera to external sync.")
        elif tr is not None and tr.hw_timed:
            # a continuous output task in step with the inputs: pulse positions are exact sample indices
            self._out_q = a.out_chunks(fs)
            self._chassis = chassis
            self._want_ai_clk = tr.ai_clocked
            self.out = self._make_out_task(fs, tr.ai_clocked)
            self.hw_trigger = True
        elif tr is not None:
            self.static = nidaqmx.Task("HammerTrigStatic")
            _add_trigger_chan(self.static, tr)
            try:
                self.static.start()
                _, idle = trigger_levels(tr)
                self.static.write(bool(idle > 0.5) if tr.kind == "DO" else float(idle))
            except Exception as e:
                self.static.close()
                self.static = None
                if _needs_clock(e):
                    raise RuntimeError(f"{tr.physical} cannot be switched on demand: its module only supports "
                                       "sample-clocked output (e.g. the delta-sigma NI 9260). Set the external "
                                       "trigger Timing to 'Hardware-timed' on the Acquisition Setup tab, or use a "
                                       "DO line.") from e
                raise
            notes.append(f"External trigger on {tr.physical} ({tr.kind}), on-demand (software-timed): the pulse "
                         "leaves within a few ms of the decision; its sample index in the log is approximate.")
        if self.cfg.null_bridges and any(c.sensor_type.startswith("Strain") for c in self.channels):
            self.ai.perform_bridge_offset_nulling_cal(skip_unsupported_channels=True)
            notes.append("Strain-bridge offsets nulled.")
        self.ai.control(K.TaskMode.TASK_COMMIT)
        if self.out is not None:
            notes += self._commit_out(fs)
        self.reader = AnalogMultiChannelReader(self.ai.in_stream)
        self._buf = np.zeros((len(self.channels), self.n))
        self.fs = fs
        return fs, notes

    def _make_out_task(self, fs: float, ai_clocked: bool):
        """Continuous trigger-output task: either on the AI sample clock, or (default, works with every
        module, as the shaker app's 'shared start trigger') on its own clock at the same rate, started
        by the AI start trigger."""
        import nidaqmx
        import nidaqmx.constants as K
        q, n = self._out_q, self.n
        task = nidaqmx.Task("HammerTrigOut")
        try:
            _add_trigger_chan(task, self.trig)
            src = f"/{self._chassis}/ai/SampleClock" if ai_clocked else ""
            task.timing.cfg_samp_clk_timing(fs, source=src, sample_mode=K.AcquisitionType.CONTINUOUS,
                                            samps_per_chan=n * 2 * q)
            task.out_stream.regen_mode = K.RegenerationMode.DONT_ALLOW_REGENERATION
            task.out_stream.output_buf_size = n * 2 * q
            if not ai_clocked:
                task.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{self._chassis}/ai/StartTrigger")
        except Exception:
            task.close()
            raise
        self._ai_clocked = ai_clocked
        return task

    def _commit_out(self, fs: float) -> List[str]:
        """Commit the output task (after the AI task, so that its signals can be routed). If the AI
        sample clock cannot be used as its clock, fall back to own clock + AI start trigger."""
        import nidaqmx.constants as K
        notes = []
        try:
            self.out.control(K.TaskMode.TASK_COMMIT)
        except Exception as e:
            if not self._ai_clocked:
                raise
            self.out.close()
            notes.append(f"Note: the trigger output cannot run on the AI sample clock here ({str(e).splitlines()[0]}). "
                         "Using its own clock at the same rate, started by the AI start trigger instead.")
            self.out = self._make_out_task(fs, False)
            self.out.control(K.TaskMode.TASK_COMMIT)
        q, tr = self._out_q, self.trig
        how = "on the AI sample clock" if self._ai_clocked else "on its own clock, started by the AI start trigger"
        notes.append(f"External trigger on {tr.physical} ({tr.kind}), hardware-timed {how}, {tr.pulse_samples}-sample "
                     f"pulses, output queued {q} reads ({q * self.n / fs * 1000:.0f} ms) ahead.")
        if not self._ai_clocked:
            f_out = float(self.out.timing.samp_clk_rate)
            if abs(f_out - fs) > 1e-9 * fs:
                notes.append(f"WARNING: trigger output rate {f_out:.9g} Hz ≠ AI rate {fs:.9g} Hz: pulse positions drift "
                             "slowly from the logged sample index. Choose a sampling rate both modules support exactly.")
        if tr.kind == "AO":
            try:
                import nidaqmx.system
                product = nidaqmx.system.Device(tr.physical.strip().lstrip("/").split("/")[0]).product_type
            except Exception:
                product = ""
            if "926" in product:
                notes.append(f"Note: {product} is a delta-sigma output module: its filter delays the pulse (logged index "
                             "+ the module's group delay) and rounds its edges. Use ≥ 10 samples, or a DO line.")
        return notes

    # ------------------------------------------------------------------ streaming
    def write_trigger(self, t: np.ndarray) -> None:
        if self.out is None:
            return
        if self.trig.kind == "AO":
            from nidaqmx.stream_writers import AnalogSingleChannelWriter
            if getattr(self, "_wr_task", None) is not self.out:
                self._wr = AnalogSingleChannelWriter(self.out.out_stream, auto_start=False)
                self._wr_task = self.out
            self._wr.write_many_sample(np.ascontiguousarray(t, dtype=np.float64), timeout=10.0)
        else:
            self.out.write((np.asarray(t) > 0.5).tolist(), auto_start=False, timeout=10.0)

    def pulse_now(self, active: bool) -> None:
        if self.static is None:
            return
        on, off = trigger_levels(self.trig)
        v = on if active else off
        self.static.write(bool(v > 0.5) if self.trig.kind == "DO" else float(v))

    def start(self) -> None:
        if self.out is not None:
            self.out.start()          # armed: waits for the AI start trigger / AI sample clock
        self.ai.start()

    def read_chunk(self) -> np.ndarray:
        self.reader.read_many_sample(self._buf, number_of_samples_per_channel=self.n, timeout=10.0)
        return self._buf * self._scale[:, None]

    def samples_acquired(self) -> int:
        try:
            return int(self.ai.in_stream.total_samp_per_chan_acquired)
        except Exception:
            return -1

    def backlog(self) -> int:
        try:
            return int(self.ai.in_stream.avail_samp_per_chan)
        except Exception:
            return 0

    def stop(self) -> None:
        for t in (self.ai, self.out, self.static):
            try:
                if t is not None:
                    t.stop()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        for t in (self.ai, self.out, self.static):
            try:
                if t is not None:
                    t.close()
            except Exception:
                pass
        self.ai = self.out = self.static = None
        self._set_trigger_idle()

    def _set_trigger_idle(self) -> None:
        """Put the trigger line at its idle level (before and after the run), so that an active-low
        pulse still gives a falling edge. Modules without on-demand output get a short clocked write."""
        tr = self.trig
        if tr is None or tr.frame_sync:
            return
        _, idle = trigger_levels(tr)
        try:
            write_trigger_samples(tr, [idle] * 4, self.cfg.acq.fs)
        except Exception:
            pass


# ----------------------------------------------------------------------------- simulator
class SimBackend:
    """Simulated impact test: a hammer (half-sine pulses, ~0.6–1.4 ms) strikes a 3-mode structure;
    accelerometer responses by modal superposition, stateful across reads. Hits come at random
    intervals (auto) and/or on request (`request_hit`). About 1 in 10 hits is a double hit and
    the occasional hard hit overloads the hammer channel."""

    def __init__(self, cfg: AppConfig, probe: bool = False, realtime: bool = True, seed: int = None,
                 auto_hits: bool = True, interval=(1.6, 3.2)):
        from scipy import signal
        self._signal = signal
        self.cfg = cfg
        self.channels = cfg.enabled_channels()
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.auto_hits = auto_hits
        self.interval = interval
        self.double_prob = 0.10
        self.overload_prob = 0.04
        self.noise_floor = {"g": 2e-4, "N": 0.05}
        t = cfg.trigger
        self.trig = t if (t.enabled and (t.in_probe or not probe)) else None
        self.hw_trigger = self.trig is not None and self.trig.hw_timed and not self.trig.frame_sync
        self.trig_queue: List[np.ndarray] = []
        self.trigger_edges: List[int] = []
        self._trig_prev = False
        self._pulses: List[Tuple[int, float, float]] = []    # (start sample, amplitude, duration s)
        self._lock = threading.Lock()
        self._hit_req = 0
        self.k = 0

    def configure(self, chunk: int) -> Tuple[float, List[str]]:
        sig = self._signal
        fs = self.cfg.acq.fs
        self.fs, self.n = fs, int(chunk)
        self.fn = np.array([0.012, 0.034, 0.068]) * fs          # 123, 348, 696 Hz at 10.24 kHz
        self.zeta = np.array([0.015, 0.012, 0.010])
        self.modes = []
        for fn, z in zip(self.fn, self.zeta):
            w = 2 * np.pi * fn
            self.modes.append(sig.bilinear([1.0, 0.0, 0.0], [1.0, 2 * z * w, w * w], fs))
        self.ref_i = self.cfg.reference_index()
        hammer_node = (self.cfg.save.point_node if self.cfg.save.roving or self.ref_i < 0
                       else self.channels[self.ref_i].node)
        self.set_point(hammer_node)
        self.zi = [[np.zeros(2) for _ in self.modes] for _ in self.channels]
        self._next_auto = int(self.fs * self.rng.uniform(*self.interval) + 1.0 * self.fs)
        notes = [f"SIMULATION: modes at {', '.join(f'{f:.1f}' for f in self.fn)} Hz "
                 f"(ζ = {', '.join(f'{100 * z:.1f}' for z in self.zeta)} %), hammer at node {hammer_node}"
                 + ("; random hits every 1.6–3.2 s" if self.auto_hits else "; hits only on request")]
        if self.trig is not None:
            if self.trig.frame_sync:
                notes.append(f"Frame sync (simulated): AI sample clock → {self.trig.physical}.")
            else:
                notes.append(f"External trigger on {self.trig.physical} ({self.trig.kind}, "
                             f"{'hardware-timed' if self.hw_trigger else 'on-demand'}), simulated.")
        return fs, notes

    def set_point(self, hammer_node: int):
        """Mode-shape gains for the hammer at `hammer_node` (beam-like shapes over nodes 0…9)."""
        shape = lambda r, node: np.sin((r + 1) * np.pi * ((int(node) % 10) + 0.5) / 10.5)
        gain = np.zeros((len(self.channels), len(self.modes)))
        for i, c in enumerate(self.channels):
            if i == self.ref_i or c.sensor_type == "Voltage":
                continue
            for r in range(len(self.modes)):
                g = shape(r, c.node) * shape(r, hammer_node) / 2.0            # modal mass 2 kg
                if c.units == "g":
                    g /= 9.80665
                elif c.sensor_type.startswith("Strain"):
                    g *= 2e-6
                gain[i, r] = g
        self.gain = gain

    def request_hit(self, amp: float = None):
        with self._lock:
            self._hit_req += 1 if amp is None else 0
            if amp is not None:
                self._pulses.append((self.k * self.n + int(0.02 * self.fs), float(amp), 0.001))

    def _schedule(self, s0: int, s1: int):
        """Add hammer pulses that start in [s0, s1)."""
        new = []
        with self._lock:
            req, self._hit_req = self._hit_req, 0
        for _ in range(req):
            new.append(s0 + int(self.rng.integers(0, max(1, s1 - s0))))
        while self.auto_hits and self._next_auto < s1:
            new.append(max(self._next_auto, s0))
            self._next_auto += int(self.fs * self.rng.uniform(*self.interval))
        for st in new:
            amp = self.rng.uniform(80, 350)
            dur = self.rng.uniform(0.0006, 0.0014)
            if self.rng.random() < self.overload_prob and self.ref_i >= 0:
                amp = 1.2 * self.channels[self.ref_i].max_range
            self._pulses.append((st, amp, dur))
            if self.rng.random() < self.double_prob:
                self._pulses.append((st + int(self.rng.uniform(0.003, 0.012) * self.fs),
                                     amp * self.rng.uniform(0.15, 0.5), dur))

    def write_trigger(self, t: np.ndarray) -> None:
        self.trig_queue.append(np.asarray(t, dtype=float))

    def pulse_now(self, active: bool) -> None:
        if active:
            self.trigger_edges.append(self.k * self.n)

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.k = 0

    def read_chunk(self) -> np.ndarray:
        s0, s1 = self.k * self.n, (self.k + 1) * self.n
        if self.hw_trigger:
            if not self.trig_queue:
                raise RuntimeError("Simulated trigger output underflow: output queue ran dry")
            tr = self.trig_queue.pop(0)
            on, _ = trigger_levels(self.trig)
            active = np.isclose(tr, on)
            starts = np.flatnonzero(active & ~np.concatenate([[self._trig_prev], active[:-1]]))
            self.trigger_edges += [int(s0 + i) for i in starts]
            self._trig_prev = bool(active[-1])
        if self.realtime:
            due = self.t0 + (self.k + 1) * self.n / self.fs
            time.sleep(max(0.0, due - time.perf_counter()))
        self._schedule(s0, s1)
        idx = np.arange(s0, s1)
        force = np.zeros(self.n)
        keep = []
        for st, amp, dur in self._pulses:
            m = int(np.ceil(dur * self.fs)) + 1
            if st + m < s0:
                continue
            keep.append((st, amp, dur))
            t = (idx - st) / self.fs
            on = (t >= 0) & (t <= dur)
            force[on] += amp * np.sin(np.pi * t[on] / dur)
        self._pulses = keep
        self.k += 1
        out = np.zeros((len(self.channels), self.n))
        for i, c in enumerate(self.channels):
            if i == self.ref_i:
                out[i] = force
                continue
            for r, (b, a) in enumerate(self.modes):
                if self.gain[i, r] != 0:
                    y, self.zi[i][r] = self._signal.lfilter(b, a, force, zi=self.zi[i][r])
                    out[i] += self.gain[i, r] * y
        for i, c in enumerate(self.channels):
            nf = self.noise_floor.get(c.units, 1e-4 * c.max_range)
            out[i] += nf * self.rng.standard_normal(self.n)
            np.clip(out[i], -c.max_range, c.max_range, out=out[i])          # ADC saturation
        return out

    def samples_acquired(self) -> int:
        return self.k * self.n

    def backlog(self) -> int:
        return 0

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_backend(cfg: AppConfig, simulate: bool, probe: bool = False, **kw):
    return SimBackend(cfg, probe=probe, **kw) if simulate else NIDAQBackend(cfg, probe=probe)
