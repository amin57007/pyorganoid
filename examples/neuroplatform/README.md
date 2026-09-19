# FinalSpark Neuroplatform closed-loop simulation

This example shows how to **practice the Neuroplatform procedure in PyOrganoid** before you have live access to FinalSpark hardware.

It does **not** drive living organoids. It copies the control loop you would write against `neuroplatformv2`:

```text
pixels → 8 electrode amplitudes → StimParam + trigger
      → 200 ms integrate-and-fire window
      → read_count()  (128 channels, 8 used)
      → decoder MLP → class 0–9
```

Full write-up of algorithm, tensors, and measured scores: [REPORT.md](REPORT.md).

## Hardware analog

| FinalSpark Neuroplatform | This simulation |
| --- | --- |
| 4 MEAs × 4 organoids × 8 electrodes = **128** channels | `NeuroplatformEnvironment.N_CHANNELS = 128` |
| **8** electrodes under one organoid | `NeuroplatformOrganoid` with 8 cells |
| Trigger keys **0–15** | length-16 `uint8` mask |
| Biphasic current stim (`StimParam`) | `StimParam.phase_amplitude1` (abstract µA) |
| `intan.send_stimparam` / `trigger.send` / `read_count` | `env.send_stimparam` / `env.send` / `env.read_count` |
| ~200 ms count window, 35–50 ms network delay | 10 inner steps (documented 200 ms + 40 ms) |
| Living tissue | 8 leaky integrate-and-fire cells, threshold 1.0 |
| Your Python model | encoder (row means) + decoder (MLP on 8 spike counts) |

Picture size in this demo is **8×8** (UCI digits, 64 pixels, values scaled to 0–1). That cannot map 1:1 onto 8 electrodes, so each electrode gets one **row mean**.

## Run

```bash
pip install -e ".[sklearn]"
python examples/neuroplatform/run_closed_loop.py
```

This trains a decoder on 797 digits, then runs **1000** closed-loop trials. Figures land in `examples/neuroplatform/figures/`.

## Minimal closed loop

```python
import numpy as np
from pyorganoid import (
    NeuroplatformEnvironment, NeuroplatformOrganoid, NeuroplatformScheduler,
)

env = NeuroplatformEnvironment(
    organoid_electrodes=NeuroplatformEnvironment.site_channels(mea=0, organoid=0),
    window_steps=10,
)
organoid = NeuroplatformOrganoid(env)
scheduler = NeuroplatformScheduler(organoid)

image = np.random.rand(8, 8)          # 8x8 handwritten-style frame
counts_128 = scheduler.present(image) # trigger + 200 ms window
counts_8 = env.organoid_counts(counts_128)
# pred = decoder.predict(counts_8.reshape(1, -1))
```

`scheduler.present` is the stand-in for:

```python
intan.send_stimparam(params)
trigger_gen.send(mask)     # length 16
time.sleep(0.2)
counts = intan.read_count()  # length 128
```

When you get a Neuroplatform token, keep the same encoder and decoder. Replace `scheduler.present` with the real `StimParam` / trigger / `read_count` calls. Docs: [stimulating](https://finalspark-np.github.io/np-docs/np_core/np_usage.html), [closed loop](https://finalspark-np.github.io/np-docs/np_core/closed_loop.html).

## What is not simulated

Biology, 6σ spike detection, 3 ms waveforms, 30 kHz sampling, electrode impedance, organoids that do not cover the pad, 10 s stim-param upload, or cameras / pumps / UV uncaging.
