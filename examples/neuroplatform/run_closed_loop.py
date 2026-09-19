"""Simulate a FinalSpark Neuroplatform closed loop on 1000 handwritten digits."""
__import__("warnings").filterwarnings("ignore")

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_TEST = 1000
WINDOW_STEPS = 10
RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(HERE, "figures")
OUTPUT_DIR = os.path.join(HERE, "output")


def load_digits(n_test=N_TEST, random_state=RANDOM_STATE):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype(float) / 16.0
    y = digits.target.astype(int)
    return train_test_split(X, y, test_size=n_test, random_state=random_state, stratify=y)


def train_decoder(counts, labels):
    from sklearn.neural_network import MLPClassifier

    decoder = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    decoder.fit(counts, labels)
    return decoder


def build_system(window_steps=WINDOW_STEPS, noise_std=0.02, seed=RANDOM_STATE):
    from pyorganoid import NeuroplatformEnvironment, NeuroplatformOrganoid, NeuroplatformScheduler

    np.random.seed(seed)
    electrodes = NeuroplatformEnvironment.site_channels(mea=0, organoid=0)
    environment = NeuroplatformEnvironment(
        organoid_electrodes=electrodes,
        window_steps=window_steps,
        noise_std=noise_std,
        window_ms=200.0,
        latency_ms=40.0,
        dimensions=2,
        size=50.0,
    )
    organoid = NeuroplatformOrganoid(environment, electrodes=electrodes, threshold=1.0, gain=1.0, leak=0.05)
    scheduler = NeuroplatformScheduler(organoid)
    return environment, organoid, scheduler


def collect_counts(scheduler, images, verbose=False):
    result = scheduler.simulate(images, decoder=None, verbose=verbose)
    return result["organoid_counts"]


def plot_confusion(y_true, y_pred, filename, dpi=200):
    matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < 10 and 0 <= p < 10:
            matrix[t, p] += 1
    accuracy = float(np.mean(y_true == y_pred))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Neuroplatform closed-loop decoding\n{len(y_true)} digits, accuracy={accuracy:.1%}")
    vmax = matrix.max() if matrix.max() else 1
    for i in range(10):
        for j in range(10):
            color = "white" if matrix[i, j] > vmax / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def plot_encoding_examples(images, y_true, y_pred, stims, counts, filename, n=8, dpi=200):
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
    if n == 1:
        axes = np.array([axes])
    for i in range(n):
        axes[i, 0].imshow(images[i].reshape(8, 8), cmap="gray_r")
        color = "#14803C" if y_true[i] == y_pred[i] else "#C0392B"
        axes[i, 0].set_title(f"true {y_true[i]} / pred {y_pred[i]}", color=color, fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].bar(range(8), stims[i], color="#1F4E79")
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title("8-electrode stim", fontsize=9)
        axes[i, 1].set_xticks(range(8))
        axes[i, 2].bar(range(8), counts[i], color="#C0392B")
        axes[i, 2].set_title("spike counts (200 ms)", fontsize=9)
        axes[i, 2].set_xticks(range(8))
    fig.suptitle("Input image → electrode amplitudes → organoid spike counts", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def plot_running_accuracy(y_true, y_pred, filename, dpi=200):
    correct = (y_true == y_pred).astype(float)
    running = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(running, color="#1F4E79", linewidth=2)
    ax.axhline(running[-1], color="#C0392B", linestyle="--", label=f"Final accuracy {running[-1]:.1%}")
    ax.set_xlabel("Closed-loop trials")
    ax.set_ylabel("Running accuracy")
    ax.set_title("Decoder accuracy on 1000 Neuroplatform trials")
    ax.set_ylim(0.0, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def plot_organoid_structure_matplotlib(organoid, filename, dpi=200):
    """Draw the 8-electrode organoid without Graphviz (works on Windows)."""
    environment = organoid.environment
    electrodes = list(getattr(environment, "organoid_electrodes", range(8)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.8, 3.2)
    ax.axis("off")
    ax.set_title("Neuroplatform organoid (8 electrodes, Graphviz not required)")

    env_box = plt.Rectangle((0.2, 2.35), 5.1, 0.7, fill=True, facecolor="#E8F1F8",
                            edgecolor="#1F4E79", linewidth=1.5)
    ax.add_patch(env_box)
    ax.text(2.75, 2.7, f"{type(environment).__name__}  |  128 ch, 16 triggers, 8-site MEA",
            ha="center", va="center", fontsize=10)

    org_box = plt.Rectangle((0.2, 1.45), 5.1, 0.7, fill=True, facecolor="#F7F3E8",
                            edgecolor="#8A6D3B", linewidth=1.5)
    ax.add_patch(org_box)
    ax.text(2.75, 1.8, f"{type(organoid).__name__}  |  {len(organoid.agents)} I&F cells",
            ha="center", va="center", fontsize=10)

    for i, cell in enumerate(organoid.agents):
        col = i % 4
        row = 1 - (i // 4)
        x, y = 0.55 + col * 1.25, 0.15 + row * 0.7
        ax.add_patch(plt.Circle((x + 0.4, y + 0.22), 0.28, facecolor="#1F4E79", edgecolor="black"))
        electrode = getattr(cell, "electrode_index", electrodes[i] if i < len(electrodes) else i)
        ax.text(x + 0.4, y + 0.22, f"e{electrode}", ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    print(f'Organoid structure plot saved as "{filename}"')


def try_plot_organoid(organoid, filename):
    """Prefer Graphviz; fall back to matplotlib if `dot` is missing (common on Windows)."""
    import shutil

    if shutil.which("dot"):
        try:
            organoid.plot_organoid(filename, show_properties=True, dpi=200)
            return
        except Exception as exc:
            print(f"Graphviz plot failed ({exc}); using matplotlib instead.")
    else:
        print(
            "Graphviz `dot` is not on PATH (optional). "
            "On Windows install https://graphviz.org/download/ and tick 'Add to PATH', "
            "or ignore this: the closed-loop figures still save via matplotlib."
        )
    plot_organoid_structure_matplotlib(organoid, filename)


def plot_mea_layout(environment, counts, filename, dpi=200):
    grid = np.zeros((2, 4), dtype=float)
    for i, value in enumerate(counts[:8]):
        row, col = divmod(i, 4)
        grid[row, col] = value
    vmax = max(float(np.max(counts[:8])), 1.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(grid, cmap="hot", vmin=0, vmax=vmax)
    fig.colorbar(image, ax=ax, label="Spike count")
    for i in range(8):
        row, col = divmod(i, 4)
        color = "black" if counts[i] > 0.55 * vmax else "white"
        ax.text(col, row, f"e{environment.organoid_electrodes[i]}\n{int(counts[i])}",
                ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Simulated 2x4 MEA site (8 electrodes under one organoid)")
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)


def run(n_test=N_TEST, verbose=True, plot=True):
    import joblib
    from pyorganoid import NeuroplatformEnvironment

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_digits(n_test=n_test)
    environment, organoid, scheduler = build_system()

    if plot:
        try_plot_organoid(organoid, os.path.join(FIGURE_DIR, "neuroplatform_organoid.png"))

    if verbose:
        print(f"Collecting spike counts on {len(X_train)} train digits (8 electrodes, {WINDOW_STEPS} inner steps)...")
    train_counts = collect_counts(scheduler, X_train, verbose=False)
    decoder = train_decoder(train_counts, y_train)
    joblib.dump(decoder, os.path.join(OUTPUT_DIR, "spike_decoder.pkl"))

    if verbose:
        print(f"Running closed loop on {len(X_test)} holdout digits...")
    test_result = scheduler.simulate(X_test, decoder=decoder, verbose=verbose)
    y_pred = test_result["predictions"]
    test_counts = test_result["organoid_counts"]
    accuracy = float(np.mean(y_pred == y_test))
    stims = np.stack([NeuroplatformEnvironment.encode_image(img) for img in X_test])

    from sklearn.metrics import classification_report

    report_text = classification_report(y_test, y_pred, digits=3)
    per_class = {}
    for digit in range(10):
        mask = y_test == digit
        per_class[str(digit)] = {
            "n": int(np.sum(mask)),
            "accuracy": float(np.mean(y_pred[mask] == y_test[mask])) if np.any(mask) else 0.0,
        }

    summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "picture_size": [8, 8],
        "n_pixels": 64,
        "n_electrodes": 8,
        "n_channels": 128,
        "n_triggers": 16,
        "window_steps": WINDOW_STEPS,
        "window_ms": 200.0,
        "latency_ms": 40.0,
        "encoding": "row_mean_8x8_to_8_electrodes",
        "tissue": "8 integrate-and-fire cells, threshold=1.0, leak=0.05",
        "decoder": "MLPClassifier(32, 16) on 8-d spike counts",
        "accuracy": accuracy,
        "per_class": per_class,
        "classification_report": report_text,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_text)

    if verbose:
        print(report_text)
        print(f"Closed-loop accuracy: {accuracy:.1%} on {len(X_test)} digits")

    if plot:
        plot_confusion(y_test, y_pred, os.path.join(FIGURE_DIR, "confusion_matrix.png"))
        plot_running_accuracy(y_test, y_pred, os.path.join(FIGURE_DIR, "running_accuracy.png"))
        plot_encoding_examples(
            X_test, y_test, y_pred, stims, test_counts,
            os.path.join(FIGURE_DIR, "encoding_examples.png"),
        )
        plot_mea_layout(environment, test_counts[0], os.path.join(FIGURE_DIR, "mea_site.png"))

    return summary


def main():
    run(n_test=N_TEST, verbose=True, plot=True)


if __name__ == "__main__":
    main()
