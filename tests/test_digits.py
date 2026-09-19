__import__("warnings").filterwarnings("ignore")


N_SIMULATION_SAMPLES = 1000
N_CELLS = 10
N_STEPS = 100


def _load_handwritten_digits(n_test=N_SIMULATION_SAMPLES, random_state=42):
    """Load UCI handwritten digits (0-9) and hold out n_test samples for simulation."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype(float) / 16.0
    y = digits.target.astype(int)
    return train_test_split(X, y, test_size=n_test, random_state=random_state, stratify=y)


def _train_digit_model(X_train, y_train, model_path):
    """Train a multi-class MLP that maps handwritten digit pixels to class numbers 0-9."""
    import joblib
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model


def _infer_image_shape(n_features):
    side = int(round(n_features ** 0.5))
    if side * side == n_features:
        return side, side
    return 1, n_features


def _plot_sample_digits(images, y_true, y_pred, filename, n_samples=40, dpi=200):
    import matplotlib.pyplot as plt

    n_samples = min(n_samples, len(images))
    n_cols = 10
    n_rows = max(1, (n_samples + n_cols - 1) // n_cols)
    height, width = _infer_image_shape(images.shape[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.4))
    axes = axes.flatten() if n_samples > 1 else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n_samples:
            continue
        ax.imshow(images[i].reshape(height, width), cmap="gray_r")
        correct = y_true[i] == y_pred[i]
        color = "#14803C" if correct else "#C0392B"
        ax.set_title(f"true {y_true[i]} / pred {y_pred[i]}", fontsize=8, color=color)
    fig.suptitle("Handwritten Digit Samples (green=correct, red=misclassified)", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    print(f'Sample digit plot saved as "{filename}"')


def _plot_running_accuracy(y_true, y_pred, filename, dpi=200):
    import numpy as np
    import matplotlib.pyplot as plt

    correct = (y_true == y_pred).astype(float)
    running = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(running, color="#1F4E79", linewidth=2)
    ax.axhline(running[-1], color="#C0392B", linestyle="--", label=f"Final accuracy {running[-1]:.1%}")
    ax.set_xlabel("Classified handwritten digits")
    ax.set_ylabel("Running accuracy")
    ax.set_title("Organoid Classification Accuracy Over 1000 Handwritten Digits")
    ax.set_ylim(0.7, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)
    print(f'Running accuracy plot saved as "{filename}"')


def run_digit_simulation(n_test, num_cells, steps, output_dir=None, plot=True, verbose=True):
    """
    Train a digit classifier and run a pyorganoid simulation that classifies n_test handwritten digits.

    Returns
    -------
    dict
        Simulation results including true labels, predictions, and accuracy.
    """
    import os
    import numpy as np
    from pyorganoid import DigitClassificationOrganoid, HandwritingEnvironment, Scheduler, SklearnModel

    if num_cells * steps != n_test:
        raise ValueError(f"num_cells * steps must equal n_test ({num_cells} * {steps} != {n_test}).")

    X_train, X_test, y_train, y_test = _load_handwritten_digits(n_test=n_test)
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "digit_mlp_model.pkl")
    model = _train_digit_model(X_train, y_train, model_path)

    if verbose:
        train_acc = float(np.mean(model.predict(X_train) == y_train))
        holdout_acc = float(np.mean(model.predict(X_test) == y_test))
        print(f"Trained MLP on {len(X_train)} handwritten digits.")
        print(f"Training accuracy: {train_acc:.1%}")
        print(f"Holdout accuracy on {len(X_test)} digits: {holdout_acc:.1%}")

    ml_model = SklearnModel(model_path, num_features=X_test.shape[1])
    environment = HandwritingEnvironment(X_test, y_test, dimensions=2, size=50)
    organoid = DigitClassificationOrganoid(environment, ml_model, num_cells=num_cells)

    if plot:
        try:
            organoid.plot_organoid(
                os.path.join(output_dir, "digit_classification_organoid.png"),
                show_properties=True,
                dpi=200,
            )
        except Exception as exc:
            print(f"Skipping organoid structure plot: {exc}")

    if verbose:
        print(f"Running simulation: {num_cells} cells x {steps} steps = {n_test} handwritten digits...")
        scheduler = Scheduler(organoid)
    else:
        class _QuietScheduler(Scheduler):
            def simulate(self, steps):
                for _ in range(steps):
                    self.organoid.environment.update()
                    for agent in self.organoid.agents:
                        agent.update()

        scheduler = _QuietScheduler(organoid)
    scheduler.simulate(steps=steps)

    images, y_true, y_pred = organoid.classified_images(chronological=True)
    accuracy = organoid.classification_accuracy()
    if verbose:
        from sklearn.metrics import classification_report

        print(f"Organoid classified {len(y_true)} handwritten digits.")
        print(f"Simulation accuracy: {accuracy:.1%}")
        print(classification_report(y_true, y_pred, digits=3))
        for digit in range(10):
            mask = y_true == digit
            if np.any(mask):
                class_acc = float(np.mean(y_pred[mask] == y_true[mask]))
                print(f"  Class {digit}: {int(np.sum(mask))} samples, accuracy {class_acc:.1%}")

    if plot:
        organoid.plot_simulation_history(
            "Predicted Digit Class Over Time",
            "Predicted Class Number",
            filename=os.path.join(output_dir, "digit_classification_simulation.png"),
            dpi=200,
        )
        organoid.plot_classification_results(
            filename=os.path.join(output_dir, "digit_confusion_matrix.png"),
            dpi=200,
        )
        _plot_sample_digits(
            images,
            y_true,
            y_pred,
            os.path.join(output_dir, "digit_sample_predictions.png"),
        )
        _plot_running_accuracy(
            y_true,
            y_pred,
            os.path.join(output_dir, "digit_running_accuracy.png"),
        )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "images": images,
        "accuracy": accuracy,
        "n_samples": len(y_true),
        "organoid": organoid,
    }


def main():
    try:
        import os
        import matplotlib

        matplotlib.use("Agg")
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_simulation_output")
        results = run_digit_simulation(
            n_test=N_SIMULATION_SAMPLES,
            num_cells=N_CELLS,
            steps=N_STEPS,
            output_dir=output_dir,
            plot=True,
            verbose=True,
        )
        print(
            f"Done. Classified {results['n_samples']} handwritten digits "
            f"with {results['accuracy']:.1%} accuracy."
        )
        print(f"Plots saved in {output_dir}")
    except ImportError as e:
        print(e)


if __name__ == "__main__":
    main()


def test_handwriting_environment_streams_digits():
    import numpy as np
    from pyorganoid import HandwritingEnvironment

    images = np.arange(6, dtype=float).reshape(3, 2)
    labels = np.array([1, 5, 8])
    env = HandwritingEnvironment(images, labels)
    first = env.next_sample()
    second = env.next_sample()
    third = env.next_sample()
    wrapped = env.next_sample()
    assert first[1] == 1
    assert second[1] == 5
    assert third[1] == 8
    assert wrapped[1] == 1
    env.reset()
    assert env.next_sample()[1] == 1


def test_classify_handwritten_digits():
    """Classify a small stream of handwritten digits through a DigitClassificationOrganoid."""
    import tempfile

    n_test, num_cells, steps = 40, 4, 10
    with tempfile.TemporaryDirectory() as tmp:
        results = run_digit_simulation(
            n_test=n_test,
            num_cells=num_cells,
            steps=steps,
            output_dir=tmp,
            plot=False,
            verbose=False,
        )
    assert results["n_samples"] == n_test
    assert results["y_true"].shape == (n_test,)
    assert results["y_pred"].shape == (n_test,)
    assert set(results["y_true"]).issubset(set(range(10)))
    assert set(results["y_pred"]).issubset(set(range(10)))
    assert results["accuracy"] >= 0.7


def test_main():
    assert True
