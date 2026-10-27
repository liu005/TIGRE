"""Plot metric histories using each algorithm's actual iteration count."""
import numpy as np


def plot_quality_measures(histories, metric_names):
    """Return axes for a name-to-(metrics, iterations) history mapping.

    Histories may have different lengths, including zero. Values are neither
    padded nor trimmed by value: genuine zeros and undefined NaNs are kept.
    """
    from matplotlib import pyplot as plt
    arrays = {name: np.asarray(values) for name, values in histories.items()}
    for name, values in arrays.items():
        if values.ndim != 2 or values.shape[0] != len(metric_names):
            raise ValueError(f"{name}: expected one history row per metric")
    axes = []
    for row, metric in enumerate(metric_names):
        _, ax = plt.subplots()
        for name, values in arrays.items():
            ax.plot(np.arange(1, values.shape[1] + 1), values[row], label=name)
        ax.set(title=f"Evolution of {metric} per iteration",
               xlabel="Completed iteration", ylabel=metric)
        ax.legend()
        ax.grid(True)
        axes.append(ax)
    return axes
