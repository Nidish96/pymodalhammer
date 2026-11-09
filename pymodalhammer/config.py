# pymodalhammer — impact-hammer modal testing with NI-DAQmx
# Copyright (C) 2026  The pymodalhammer authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License (LICENSE) for more details.

"""Configuration data classes (channels / acquisition / external trigger / saving) with JSON load & save."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import List

# ----------------------------------------------------------------------------- channels (as in the shaker app)
SENSOR_TYPES = [
    "Accelerometer (IEPE)",
    "Force (IEPE)",
    "Strain gauge (bridge)",
    "Microphone (IEPE)",
    "Pressure (IEPE)",
    "Voltage",
    "Custom",
]
CUSTOM_TYPE = "Custom"
CUSTOM_UNITS = ["EU", "mm", "µm", "m", "mm/s", "m/s", "m/s²", "kPa", "N·m", "°C", "V"]
UNITS_BY_TYPE = {
    "Accelerometer (IEPE)": ["g", "m/s²"],
    "Force (IEPE)": ["N", "lbf"],
    "Strain gauge (bridge)": ["strain"],
    "Microphone (IEPE)": ["Pa"],
    "Pressure (IEPE)": ["kPa", "Pa", "bar", "psi", "MPa"],
    "Voltage": ["V"],
    "Custom": CUSTOM_UNITS,
}


def voltage_scaled(sensor_type: str) -> bool:
    """Types read as a (optionally IEPE-powered) voltage and scaled in software by mV/unit."""
    return sensor_type in ("Voltage", CUSTOM_TYPE) or sensor_type.startswith("Pressure")


DIRECTIONS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
DIRECTION_CODE = {"+X": 1, "-X": -1, "+Y": 2, "-Y": -2, "+Z": 3, "-Z": -3}
BRIDGE_TYPES = [
    "Full bridge I", "Full bridge II", "Full bridge III",
    "Half bridge I", "Half bridge II", "Quarter bridge I", "Quarter bridge II",
]


def sensitivity_units(sensor_type: str, eu: str) -> str:
    if sensor_type.startswith("Accelerometer"):
        return "mV/g"
    if sensor_type.startswith("Force"):
        return "mV/lbf" if eu == "lbf" else "mV/N"
    if sensor_type.startswith("Microphone"):
        return "mV/Pa"
    if sensor_type.startswith("Strain"):
        return "—"
    return f"mV/{eu or 'EU'}"


TYPE_DEFAULTS = {
    #                        sens   range  iepe  coupling
    "Accelerometer (IEPE)": (100.0, 50.0, 2.0, "AC"),
    "Force (IEPE)": (2.25, 2000.0, 2.0, "AC"),        # typical impact hammer (e.g. 2.25 mV/N)
    "Strain gauge (bridge)": (0.0, 0.001, 0.0, "DC"),
    "Microphone (IEPE)": (50.0, 100.0, 2.0, "AC"),
    "Pressure (IEPE)": (10.0, 100.0, 2.0, "AC"),
    "Voltage": (1000.0, 5.0, 0.0, "Default"),
    "Custom": (100.0, 10.0, 0.0, "Default"),
}

# ----------------------------------------------------------------------------- acquisition
TRIGGER_SLOPES = ["Rising (x ≥ level)", "Falling (x ≤ −level)", "Either (|x| ≥ level)"]
FORCE_WINDOWS = ["Force window (boxcar + cosine taper)", "Rectangular (none)"]
RESP_WINDOWS = ["Exponential", "Rectangular (none)"]
ACCEPT_MODES = ["Automatic (the quality checks decide)", "Manual: I review every hit"]
DONE_ACTIONS = [
    "Stop, let me review, then I save",
    "Stop and save automatically",
    "Save, then re-arm for the next point (roving hammer)",
]
CHECK_ACTIONS = ["Reject automatically", "Ask me", "Accept, but flag it", "Don't check"]
REJECT, ASK, FLAG, NOCHECK = CHECK_ACTIONS
ESTIMATORS = ["H1", "H2", "Hv"]

# ----------------------------------------------------------------------------- external trigger
TRIGGER_MODES = [
    "Once, when the acquisition starts",
    "On every detected hit (as soon as it is detected)",
    "On every accepted hit (after the checks / your decision)",
    "On the first hit of each measurement point",
    "When the averages of a point are complete",
    "Frame sync: one pulse per sample (AI sample clock → PFI)",
]
FRAME_SYNC = TRIGGER_MODES[5]
TRIGGER_MODE_TIPS = [
    "One pulse at sample 0, i.e. when the input task starts (sample-exact).",
    "One pulse for every hit that crosses the trigger level, sent as soon as the software sees it "
    "(before the record is complete and before any check). Use it with a camera that keeps a "
    "pre-trigger (ring) buffer; the exact delay hit → pulse is written to the trigger log.",
    "One pulse per hit that is kept: after the post-trigger part of the record has arrived and the "
    "checks passed (automatic mode), or when you click Accept (manual review). Good to tell a "
    "camera with a long ring buffer to save only good hits.",
    "One pulse at the first detected hit after you arm a new measurement point.",
    "One pulse when the last average of a point has been accepted, e.g. to tell a robot or an "
    "automatic hammer to move to the next point.",
    "Routes the input task's sample clock to a PFI pin: one pulse per sample from sample 0 to the end. "
    "Drive the camera's external-sync (frame clock) input with it: frame f = sample f, no drift.",
]
TRIGGER_TIMING = ["Hardware-timed (sample-exact position, queued)", "On-demand (software write, lowest latency)"]
TRIGGER_POLARITY = ["Active high (rising edge)", "Active low (falling edge)"]
TRIGGER_SYNC = ["Own clock, started by the AI start trigger", "Clocked by the AI sample clock"]

POINT_MODES = ["Roving hammer: the hammer node is the measurement point",
               "Fixed hammer: nodes come from the channel table"]


def output_kind(physical: str) -> str:
    """'AO' for analog-output channels, 'DO' for digital lines, 'PFI' for terminals, '' if unknown."""
    p = physical.lower()
    if "pfi" in p:
        return "PFI"
    if "/ao" in p:
        return "AO"
    if "/port" in p or "/line" in p:
        return "DO"
    return ""


# ----------------------------------------------------------------------------- dataclasses
@dataclass
class ChannelConfig:
    enabled: bool = True
    physical: str = "cDAQ1Mod1/ai0"
    name: str = "Ch0"
    node: int = 1
    direction: str = "+Z"
    sensor_type: str = "Accelerometer (IEPE)"
    sensitivity: float = 100.0
    units: str = "g"
    max_range: float = 50.0
    iepe_ma: float = 2.0
    coupling: str = "AC"
    is_reference: bool = False          # the hammer force channel
    bridge: str = "Quarter bridge I"
    gage_factor: float = 2.0
    excit_v: float = 2.5
    nominal_r: float = 350.0
    poisson: float = 0.3
    custom_type: str = ""

    @property
    def type_label(self) -> str:
        return (self.custom_type or "Custom") if self.sensor_type == CUSTOM_TYPE else self.sensor_type

    @property
    def sens_units(self) -> str:
        return sensitivity_units(self.sensor_type, self.units)


@dataclass
class AcquisitionConfig:
    fs: float = 10240.0                 # Hz
    block_size: int = 8192              # samples per hit record
    pretrigger_pct: float = 5.0         # % of the record before the trigger point
    trig_channel: str = ""              # channel name; '' = the reference (hammer) channel
    trig_level: float = 20.0            # in the trigger channel's units
    trig_slope: str = TRIGGER_SLOPES[0]
    rearm_pct: float = 50.0             # re-arm when the signal falls below this % of the level
    holdoff_s: float = 0.0              # extra dead time after each record
    settle_s: float = 2.0               # triggers ignored for this long after the start (IEPE / AC settling)
    remove_offset: bool = True          # subtract each channel's pre-trigger mean
    chunk_ms: float = 50.0              # read size (latency of detection, GUI rate)
    out_queue_ms: float = 500.0         # hardware-timed trigger output: time queued ahead (slack for PC stalls)
    # windows
    force_window: str = FORCE_WINDOWS[0]
    force_width_ms: float = 10.0        # window = 1 from the record start to trigger + width
    force_taper_ms: float = 2.0         # half-cosine roll-off after the width
    resp_window: str = RESP_WINDOWS[0]
    exp_end_pct: float = 5.0            # exponential window: value at the end of the record (% of 1)
    # averaging and acceptance
    n_averages: int = 5
    accept_mode: str = ACCEPT_MODES[0]
    when_done: str = DONE_ACTIONS[0]
    # checks
    overload_pct: float = 98.0
    overload_action: str = REJECT
    double_hit_pct: float = 10.0        # a second force peak above this % of the first = double hit
    double_hit_action: str = REJECT
    force_min: float = 50.0
    force_max: float = 1000.0
    force_action: str = NOCHECK
    decay_pct: float = 5.0              # response still above this % of its peak at the record end
    decay_action: str = FLAG
    # probe mode
    probe_headroom: float = 2.0         # suggested range = headroom × largest peak seen
    probe_trig_pct: float = 10.0        # suggested trigger level = this % of the median hammer peak

    @property
    def df(self) -> float:
        return self.fs / self.block_size

    @property
    def record_s(self) -> float:
        return self.block_size / self.fs

    @property
    def pre_samples(self) -> int:
        return int(round(self.block_size * self.pretrigger_pct / 100.0))

    @property
    def manual(self) -> bool:
        return self.accept_mode == ACCEPT_MODES[1]

    def chunk_samples(self, fs: float = None) -> int:
        fs = fs or self.fs
        return int(max(16, round((self.chunk_ms / 1000.0) * fs)))

    def out_chunks(self, fs: float = None) -> int:
        """Reads' worth of trigger output queued ahead (at least 2)."""
        fs = fs or self.fs
        c = self.chunk_samples(fs) / fs * 1000.0
        return int(max(2, -(-self.out_queue_ms // c)))


@dataclass
class TriggerConfig:
    enabled: bool = False
    physical: str = "cDAQ1Mod4/port0/line0"
    mode: str = TRIGGER_MODES[1]
    timing: str = TRIGGER_TIMING[0]
    pulse_samples: int = 10
    level_v: float = 3.3
    polarity: str = TRIGGER_POLARITY[0]
    in_probe: bool = False                  # also send pulses in probe mode
    sync: str = TRIGGER_SYNC[0]             # hardware-timed output: how it is locked to the inputs

    @property
    def kind(self) -> str:
        return output_kind(self.physical)

    @property
    def frame_sync(self) -> bool:
        return self.mode == FRAME_SYNC

    @property
    def hw_timed(self) -> bool:
        return self.timing == TRIGGER_TIMING[0]

    @property
    def ai_clocked(self) -> bool:
        return self.sync == TRIGGER_SYNC[1]

    def mode_index(self) -> int:
        return TRIGGER_MODES.index(self.mode) if self.mode in TRIGGER_MODES else 0


@dataclass
class SaveConfig:
    folder: str = ""
    basename: str = "impact_test"
    save_uff: bool = True
    uff_h1: bool = True
    uff_h2: bool = False
    uff_hv: bool = False
    uff_coherence: bool = True
    uff_auto: bool = False              # averaged auto-spectra Gxx/Gyy (UFF 58 type 2)
    uff_time: bool = False              # accepted hits as time responses (UFF 58 type 1)
    save_hits_tdms: bool = True         # accepted (averaged) hits → TDMS
    save_rejected: bool = True          # also rejected / removed hits (marked in the properties)
    save_stream_tdms: bool = False      # the complete continuous record → TDMS
    point_mode: str = POINT_MODES[0]
    point_node: int = 1
    point_dir: str = "-Z"
    auto_increment: bool = True
    increment: int = 1

    @property
    def roving(self) -> bool:
        return self.point_mode == POINT_MODES[0]


@dataclass
class AppConfig:
    channels: List[ChannelConfig] = field(default_factory=list)
    acq: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    null_bridges: bool = True

    # ---------------------------------------------------------------- helpers
    def enabled_channels(self) -> List[ChannelConfig]:
        return [c for c in self.channels if c.enabled]

    def reference_index(self) -> int:
        for i, c in enumerate(self.enabled_channels()):
            if c.is_reference:
                return i
        return -1

    def trigger_index(self) -> int:
        """Index (within enabled channels) of the trigger channel (default: the reference)."""
        name = self.acq.trig_channel.strip()
        if name:
            for i, c in enumerate(self.enabled_channels()):
                if c.name == name:
                    return i
        return self.reference_index()

    def response_count(self) -> int:
        """Enabled channels other than the hammer (reference)."""
        return sum(1 for c in self.enabled_channels() if not c.is_reference)

    @property
    def time_only(self) -> bool:
        """Only the hammer channel is enabled: the hits are recorded as time data, no FRFs."""
        return self.reference_index() >= 0 and self.response_count() == 0

    def labelled_channels(self) -> List[ChannelConfig]:
        """Enabled channels with the measurement point applied: in roving-hammer mode the reference
        channel's node and direction are the point's."""
        out = []
        for c in self.enabled_channels():
            c2 = ChannelConfig(**asdict(c))
            if c2.is_reference and self.save.roving:
                c2.node, c2.direction = int(self.save.point_node), self.save.point_dir
            out.append(c2)
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "AppConfig":
        d = json.loads(text)

        def build(dc, data):
            names = {f.name for f in fields(dc)}
            return dc(**{k: v for k, v in (data or {}).items() if k in names})

        return cls(channels=[build(ChannelConfig, c) for c in d.get("channels", [])],
                   acq=build(AcquisitionConfig, d.get("acq")),
                   trigger=build(TriggerConfig, d.get("trigger")),
                   save=build(SaveConfig, d.get("save")),
                   null_bridges=d.get("null_bridges", True))

    def validate(self, probe: bool = False) -> List[str]:
        """Human-readable problems (empty = OK). Probe mode needs no reference. Without response channels
        the measurement records time data only (see `time_only`)."""
        errs = []
        ch = self.enabled_channels()
        a = self.acq
        if not ch:
            errs.append("No input channels are enabled.")
        refs = [c for c in ch if c.is_reference]
        if not probe:
            if len(refs) != 1:
                errs.append("Exactly one enabled channel must be marked as the Reference (the hammer force).")
        phys = [c.physical.strip() for c in ch]
        dup = {p for p in phys if phys.count(p) > 1}
        if dup:
            errs.append(f"Physical channel used more than once: {', '.join(sorted(dup))}")
        if any(not p for p in phys):
            errs.append("A channel has no physical channel name.")
        names = [c.name for c in ch]
        dupn = {n for n in names if names.count(n) > 1}
        if dupn:
            errs.append(f"Channel name used more than once: {', '.join(sorted(dupn))}")
        for c in ch:
            if c.sensor_type != "Strain gauge (bridge)" and c.sensitivity <= 0:
                errs.append(f"{c.name}: sensitivity must be > 0.")
            if c.sensor_type == CUSTOM_TYPE and not c.custom_type.strip():
                errs.append(f"{c.name}: enter a name for the custom sensor type.")
            if c.max_range <= 0:
                errs.append(f"{c.name}: input range must be > 0.")
        if a.fs <= 0:
            errs.append("Sampling rate must be > 0.")
        if a.block_size < 64 or a.block_size % 2:
            errs.append("Record length must be an even number of samples ≥ 64.")
        if not 0 <= a.pretrigger_pct < 50:
            errs.append("Pre-trigger must be between 0 and 50 % of the record.")
        if a.trig_channel.strip() and a.trig_channel.strip() not in names:
            errs.append(f"Trigger channel '{a.trig_channel}' is not an enabled channel.")
        ti = self.trigger_index()
        if ti < 0 and ch:
            errs.append("Choose a trigger channel (or mark the hammer as Reference).")
        elif ch and ti >= 0:
            c = ch[ti]
            if a.trig_level <= 0:
                errs.append("Trigger level must be > 0 (the slope sets the sign).")
            elif a.trig_level >= c.max_range:
                errs.append(f"Trigger level {a.trig_level:g} {c.units} is above the range of {c.name} (±{c.max_range:g}).")
        if not 0 <= a.rearm_pct < 100:
            errs.append("Re-arm level must be between 0 and 100 % of the trigger level.")
        if a.chunk_ms < 5 or a.chunk_ms > 1000:
            errs.append("Read size must be between 5 and 1000 ms.")
        if not probe:
            if a.n_averages < 1:
                errs.append("Number of averages must be ≥ 1.")
            if a.force_window == FORCE_WINDOWS[0] and not self.time_only:
                pre_ms = a.pre_samples / a.fs * 1000 if a.fs > 0 else 0
                if a.force_width_ms <= 0 or pre_ms + a.force_width_ms + a.force_taper_ms > a.record_s * 1000:
                    errs.append("Force window: the width (+ taper) must be > 0 and fit in the record.")
            if a.resp_window == RESP_WINDOWS[0] and not self.time_only and not 0.01 <= a.exp_end_pct <= 100:
                errs.append("Exponential window: the end value must be between 0.01 and 100 %.")
            if a.force_action != NOCHECK and not 0 <= a.force_min < a.force_max:
                errs.append("Force-level check: need 0 ≤ minimum < maximum.")
        t = self.trigger
        if t.enabled and (not probe or t.in_probe):
            if t.frame_sync:
                if t.kind != "PFI":
                    errs.append(f"Frame sync needs a PFI terminal (e.g. /cDAQ1Mod4/PFI0), not '{t.physical}'.")
            else:
                if t.kind == "PFI":
                    errs.append("A PFI terminal can only be used for frame sync. For pulses choose an AO channel "
                                "or a DO line (…/port0/lineN).")
                elif not t.kind:
                    errs.append(f"Trigger channel '{t.physical}' is not recognised as an analog output (…/aoN) "
                                "or a digital line (…/portN/lineM).")
                if t.pulse_samples < 1:
                    errs.append("Trigger pulse width must be at least 1 sample.")
                if t.hw_timed and a.fs > 0 and t.pulse_samples >= a.chunk_samples() * a.out_chunks():
                    errs.append("Hardware-timed trigger: the pulse must be shorter than the output queue.")
                if t.hw_timed and not 50 <= a.out_queue_ms <= 10000:
                    errs.append("Hardware-timed trigger: the output queue must be between 50 and 10000 ms.")
                if t.kind == "AO" and not 0 < t.level_v <= 10:
                    errs.append("Analog trigger level must be between 0 and 10 V.")
        if not probe and wants_folder(self) and not self.save.folder.strip():
            errs.append("Choose an output folder on the Measurement tab.")
        return errs


def wants_folder(cfg: AppConfig) -> bool:
    """A saved point always writes setup.json and the hit log, so a folder is always needed."""
    return True


def default_config() -> AppConfig:
    cfg = AppConfig()
    cfg.channels = [
        ChannelConfig(physical="cDAQ1Mod1/ai0", name="Hammer", node=1, direction="-Z",
                      sensor_type="Force (IEPE)", sensitivity=2.25, units="N", max_range=2000.0,
                      is_reference=True),
        ChannelConfig(physical="cDAQ1Mod1/ai1", name="Acc_1", node=1, direction="+Z"),
        ChannelConfig(physical="cDAQ1Mod1/ai2", name="Acc_2", node=5, direction="+Z"),
        ChannelConfig(physical="cDAQ1Mod1/ai3", name="Acc_3", node=9, direction="+Z"),
    ]
    return cfg
