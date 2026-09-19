__import__("warnings").filterwarnings("ignore")

import numpy as np


def test_site_channels_cover_128():
    from pyorganoid import NeuroplatformEnvironment

    channels = []
    for mea in range(4):
        for organoid in range(4):
            channels.extend(NeuroplatformEnvironment.site_channels(mea, organoid))
    assert channels == list(range(128))


def test_encode_row_means():
    from pyorganoid import NeuroplatformEnvironment

    image = np.zeros((8, 8), dtype=float)
    image[0] = 1.0
    image[7] = 0.5
    amplitudes = NeuroplatformEnvironment.encode_image(image)
    assert amplitudes.shape == (8,)
    assert amplitudes[0] == 1.0
    assert amplitudes[7] == 0.5
    assert amplitudes[1] == 0.0


def test_trigger_mask_and_read_count():
    from pyorganoid import NeuroplatformEnvironment, NeuroplatformOrganoid, NeuroplatformScheduler

    np.random.seed(0)
    env = NeuroplatformEnvironment(window_steps=8, noise_std=0.0)
    organoid = NeuroplatformOrganoid(env, threshold=1.0, gain=1.0, leak=0.0)
    scheduler = NeuroplatformScheduler(organoid)

    bright = np.ones(64, dtype=float)
    counts = scheduler.present(bright)
    assert counts.shape == (128,)
    assert int(counts[8:].sum()) == 0
    assert int(counts[:8].sum()) > 0

    dark = np.zeros(64, dtype=float)
    quiet = scheduler.present(dark)
    assert int(quiet[:8].sum()) == 0


def test_stimparam_rejects_bad_index():
    from pyorganoid import StimParam

    try:
        StimParam(index=200)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_closed_loop_digit_decoding():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from pyorganoid import NeuroplatformEnvironment, NeuroplatformOrganoid, NeuroplatformScheduler

    np.random.seed(42)
    digits = load_digits()
    X = digits.data.astype(float) / 16.0
    y = digits.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=80, train_size=200, random_state=42, stratify=y
    )
    env = NeuroplatformEnvironment(window_steps=10, noise_std=0.0)
    organoid = NeuroplatformOrganoid(env, threshold=1.0, gain=1.0, leak=0.05)
    scheduler = NeuroplatformScheduler(organoid)
    train_counts = scheduler.simulate(X_train, verbose=False)["organoid_counts"]
    decoder = MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=42)
    decoder.fit(train_counts, y_train)
    result = scheduler.simulate(X_test, decoder=decoder, verbose=False)
    accuracy = float(np.mean(result["predictions"] == y_test))
    assert result["counts"].shape == (80, 128)
    assert result["organoid_counts"].shape == (80, 8)
    assert accuracy >= 0.35
