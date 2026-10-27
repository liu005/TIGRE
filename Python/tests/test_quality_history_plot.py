import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tigre.utilities.visualization.plot_quality_measures import plot_quality_measures


def test_plots_independent_lengths_preserving_zeros_and_nans():
    histories = {'SIRT': np.zeros((2, 8)), 'SART': np.ones((2, 3)),
                 'OS-SART': np.full((2, 11), 2.), 'empty': np.empty((2, 0))}
    histories['SART'][1, 0] = np.nan
    axes = plot_quality_measures(histories, ['RMSE', 'CC'])
    try:
        for row, ax in enumerate(axes):
            for line in ax.lines:
                values = histories[line.get_label()][row]
                np.testing.assert_array_equal(line.get_xdata(), np.arange(1, len(values)+1))
                np.testing.assert_array_equal(line.get_ydata(), values)
            ax.figure.canvas.draw()
    finally:
        plt.close('all')


def test_rejects_inconsistent_metric_rows():
    with pytest.raises(ValueError, match='one history row'):
        plot_quality_measures({'SART': np.ones((1, 8))}, ['RMSE', 'CC'])
