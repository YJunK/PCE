import numpy as np

from topological_segmentation.topology.filtrations import probability_filtration


def test_probability_filtration_flips_probability():
    prob = np.array([[0.0, 0.5], [1.0, 0.2]])
    filt = probability_filtration(prob)
    assert np.allclose(filt, np.array([[1.0, 0.5], [0.0, 0.8]]))
