# Neuroplatform closed-loop report

Simulation of running a digit model as you would on FinalSpark Neuroplatform: stimulate 8 electrodes, wait a 200 ms window, read spike counts, decode a class.

This is **not** the 96.7% pixel-MLP organoid. That path skips the MEA. Here the tissue only sees 8 currents, matching one Neuroplatform site.

## Algorithm

Three stages. Only the middle stage would be living tissue on hardware.

### 1. Encoder (Python)

Input image \(X \in [0,1]^{8 \times 8}\). Electrode \(i \in \{0,\ldots,7\}\) receives the mean of row \(i\):

\[
a_i = \frac{1}{8}\sum_{c=0}^{7} X_{i,c}
\]

Those eight amplitudes become `StimParam.phase_amplitude1` on channels 0–7, all bound to trigger key 0. A 16-d trigger mask with bit 0 set fires the pattern and starts the count window.

### 2. Tissue (simulated organoid)

Eight leaky integrate-and-fire cells, one per electrode:

\[
V_i \leftarrow (1-\lambda)V_i + g\,a_i + \mathcal{N}(0,\sigma^2)
\]

If \(V_i \ge \theta\), count a spike on that channel and reset \(V_i \leftarrow 0\).

| Symbol | Value |
| --- | --- |
| Leak \(\lambda\) | 0.05 |
| Gain \(g\) | 1.0 |
| Noise \(\sigma\) | 0.02 |
| Threshold \(\theta\) | 1.0 |
| Inner steps \(T\) | 10 (stand-in for ~200 ms) |

Unused channels 8–127 stay at count 0, as they would if you only wired one organoid site.

### 3. Decoder (the “model”)

Let \(c \in \mathbb{N}^{8}\) be the organoid slice of `read_count()`. An MLP maps counts to class:

```text
c (8) → Dense(32, ReLU) → Dense(16, ReLU) → 10-way class
```

Trained with Adam on 797 train trials. The organoid weights are **not** trained. That matches wetware use: you program stimulation and fit a readout, you do not backprop through the dish.

Closed loop for each test image:

1. Encode row means → 8 amplitudes.
2. `arm_stim` / `send` trigger 0.
3. 10 inner cell updates.
4. `read_count()` → 128-d vector, slice 8 electrodes.
5. MLP predicts `0–9`.

## Input

| Field | Spec |
| --- | --- |
| Dataset | UCI handwritten digits (`sklearn.datasets.load_digits`) |
| Picture size | **8 × 8** pixels |
| Flattened size | 64 |
| Pixel range | 0–16 raw, used as `/16` → **[0, 1]** |
| Classes | 10 (`0–9`) |
| Train / test | 797 / **1000**, stratified, `random_state=42` |
| Stimulation | 8 row-mean amplitudes in [0, 1] |
| Trigger | `uint8[16]`, key 0 = 1 |
| MEA map | 128 channels; site 0 uses electrodes **0–7** |

## Output

| Field | Spec |
| --- | --- |
| `read_count()` | length-**128** integer spike counts |
| Organoid readout | length-**8** slice on electrodes 0–7 |
| Typical counts | about 0–5 spikes per electrode per window |
| Decoder output | class number `0–9` |
| Logs | true label, predicted label, 8 stim amplitudes, 8 counts |

Figures (this folder):

- `figures/encoding_examples.png` — image, 8 stim bars, 8 count bars
- `figures/mea_site.png` — 2×4 electrode layout for one trial
- `figures/confusion_matrix.png` — true vs predicted class
- `figures/running_accuracy.png` — accuracy over 1000 trials

## Results

Holdout: **1000** digits. Closed-loop accuracy: **65.9%** (659 / 1000). Chance is 10%. Direct-pixel MLP on the same split was 96.7%; the drop is the 64→8 electrode bottleneck, which is the point of this procedure.

| Class | n | Accuracy |
| --- | --- | --- |
| 0 | 99 | 51.5% |
| 1 | 101 | 42.6% |
| 2 | 98 | 71.4% |
| 3 | 102 | 56.9% |
| 4 | 101 | 86.1% |
| 5 | 101 | 58.4% |
| 6 | 101 | 88.1% |
| 7 | 100 | 80.0% |
| 8 | 97 | 50.5% |
| 9 | 100 | 73.0% |

Classification report:

```
              precision    recall  f1-score   support

           0      0.472     0.515     0.493        99
           1      0.558     0.426     0.483       101
           2      0.769     0.714     0.741        98
           3      0.558     0.569     0.563       102
           4      0.879     0.861     0.870       101
           5      0.641     0.584     0.611       101
           6      0.840     0.881     0.860       101
           7      0.816     0.800     0.808       100
           8      0.412     0.505     0.454        97
           9      0.689     0.730     0.709       100

    accuracy                          0.659      1000
```

Digits with a stable row-energy profile (4, 6, 7) decode well. 0 / 1 / 8 collide because row means throw away column structure (a `0` and an `8` can look similar as 8 horizontal averages). That is expected with only 8 stimulation sites.

## How this maps to a real Neuroplatform run

Keep the encoder and the fitted decoder. Replace the tissue call:

```python
# simulation
counts = scheduler.present(image)

# hardware
params[i].index = exp.electrodes[i]
params[i].phase_amplitude1 = amplitude_uA[i]
intan.send_stimparam(params)       # ~10 s, once per pattern set
trigger_gen.send(mask)             # 16 uint8
time.sleep(0.2)
counts = intan.read_count()        # 128 ints
pred = decoder.predict(counts[organoid_channels].reshape(1, -1))
```

Re-fit the decoder on live `read_count()` vectors. Do not expect 65.9% to transfer: real spike detection is 6σ, pulses are µA/µs biphasic, and the organoid may not cover every pad.

## Reproduce

```bash
python examples/neuroplatform/run_closed_loop.py
```
