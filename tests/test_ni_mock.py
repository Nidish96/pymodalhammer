# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""NI backend control flow against a mock nidaqmx that behaves like a cDAQ with a delta-sigma AO
module (NI 9260): no on-demand (static) writes (-201087) and no output on the AI sample clock
(-200414). Run: python tests/test_ni_mock.py"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nidaqmx.constants as K                                                     # noqa: E402  (pure Python)


class DaqError(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.error_code = code


LOG = []


class _Chan:
    ai_coupling = None


class _Coll:
    def __init__(self, task, kind):
        self.task, self.kind = task, kind

    def __getattr__(self, name):
        def add(phys, *a, **k):
            self.task.chans.append((self.kind, phys))
            return _Chan()
        return add


class _Timing:
    def __init__(self, task):
        self.task = task
        self.samp_clk_rate = 0.0

    def cfg_samp_clk_timing(self, rate, source="", sample_mode=None, samps_per_chan=1000):
        self.task.clocked = True
        self.task.source = source
        self.samp_clk_rate = rate


class _Start:
    def __init__(self, task):
        self.task = task

    def cfg_dig_edge_start_trig(self, src):
        self.task.start_trig = src


class Task:
    def __init__(self, name=""):
        self.name, self.chans, self.clocked, self.source, self.start_trig = name, [], False, "", ""
        self.ai_channels = _Coll(self, "ai")
        self.ao_channels = _Coll(self, "ao")
        self.do_channels = _Coll(self, "do")
        self.timing = _Timing(self)
        self.triggers = types.SimpleNamespace(start_trigger=_Start(self))
        self.out_stream = types.SimpleNamespace(regen_mode=None, output_buf_size=0)
        self.in_stream = types.SimpleNamespace(avail_samp_per_chan=0, total_samp_per_chan_acquired=0)
        self.export_signals = types.SimpleNamespace(export_signal=lambda *a: None)
        self.written = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _ao_9260(self):
        return any(k == "ao" for k, _ in self.chans)

    def control(self, mode):
        if self.source.endswith("ai/SampleClock") and self._ao_9260():
            raise DaqError("Requested sample clock source is invalid.", -200414)
        LOG.append(("commit", self.name, self.source, self.start_trig))

    def write(self, data, auto_start=False, timeout=10.0):
        if self._ao_9260() and not self.clocked:
            raise DaqError("Task contains physical channels on one or more devices that require you to specify "
                           "the Sample Clock rate.", -201087)
        self.written.append(data)
        LOG.append(("write", self.name, data if np.ndim(data) == 0 else len(data)))

    def start(self):
        if self._ao_9260() and not self.clocked:
            raise DaqError("needs sample clock", -201087)
        LOG.append(("start", self.name))

    def wait_until_done(self, timeout=10):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def perform_bridge_offset_nulling_cal(self, **k):
        pass


class Reader:
    def __init__(self, stream):
        pass

    def read_many_sample(self, buf, number_of_samples_per_channel, timeout):
        buf[:] = 0.0


UNDERFLOW_AFTER = [None]


class Writer:
    def __init__(self, stream, auto_start=False):
        pass

    def write_many_sample(self, x, timeout=10.0):
        LOG.append(("ao-chunk", len(x)))
        if UNDERFLOW_AFTER[0] is not None:
            UNDERFLOW_AFTER[0] -= 1
            if UNDERFLOW_AFTER[0] < 0:
                raise DaqError("Onboard device memory underflow.", -200621)


mod = types.ModuleType("nidaqmx")
mod.Task = Task
mod.constants = K
mod.system = types.ModuleType("nidaqmx.system")
mod.system.Device = lambda name: (_ for _ in ()).throw(RuntimeError("no device"))
mod.stream_readers = types.SimpleNamespace(AnalogMultiChannelReader=Reader)
mod.stream_writers = types.SimpleNamespace(AnalogSingleChannelWriter=Writer)
sys.modules.update({"nidaqmx": mod, "nidaqmx.system": mod.system, "nidaqmx.constants": K,
                    "nidaqmx.stream_readers": mod.stream_readers, "nidaqmx.stream_writers": mod.stream_writers})

from pymodalhammer.backends import NIDAQBackend, send_test_pulse                   # noqa: E402
from pymodalhammer.config import TRIGGER_SYNC, TRIGGER_TIMING, default_config      # noqa: E402


def cfg_ao(**kw):
    c = default_config()
    t = c.trigger
    t.enabled, t.physical, t.level_v = True, "cDAQ1Mod2/ao1", 3.3
    for k, v in kw.items():
        setattr(t, k, v)
    return c


def test_test_pulse_on_9260():
    LOG.clear()
    msg = send_test_pulse(cfg_ao().trigger, 10240.0)
    assert "sample-clocked" in msg, msg
    print("  ", msg)


def test_default_sync_start_trigger():
    LOG.clear()
    be = NIDAQBackend(cfg_ao())
    fs, notes = be.configure(512)
    out = be.out
    assert out.source == "" and out.start_trig == "/cDAQ1/ai/StartTrigger", (out.source, out.start_trig)
    assert be.hw_trigger
    be.write_trigger(np.zeros(512))
    be.start()
    print("  ", [n for n in notes if "External trigger" in n][0])


def test_ai_clock_falls_back():
    LOG.clear()
    be = NIDAQBackend(cfg_ao(sync=TRIGGER_SYNC[1]))
    fs, notes = be.configure(512)
    assert be.out.start_trig.endswith("ai/StartTrigger") and not be.out.source
    assert any("cannot run on the AI sample clock" in n for n in notes), notes
    print("  ", [n for n in notes if n.startswith("Note: the trigger")][0][:110], "…")


def test_on_demand_clear_error():
    be = NIDAQBackend(cfg_ao(timing=TRIGGER_TIMING[1]))
    try:
        be.configure(512)
    except RuntimeError as e:
        assert "Hardware-timed" in str(e)
        print("  ", str(e)[:100], "…")
        return
    raise AssertionError("expected a clear error")


def test_underflow_message():
    from pymodalhammer.engine import Callbacks, HammerEngine
    cfg = cfg_ao()
    q = cfg.acq.out_chunks()
    assert q == 10, q                                    # 500 ms of 50 ms reads
    UNDERFLOW_AFTER[0] = q + 3
    eng = HammerEngine(cfg, NIDAQBackend(cfg), Callbacks())
    try:
        eng.run()
    except RuntimeError as e:
        assert "Output queue" in str(e)
        print("  ", str(e)[:120], "…")
        return
    finally:
        UNDERFLOW_AFTER[0] = None
    raise AssertionError("expected the underflow advice")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
