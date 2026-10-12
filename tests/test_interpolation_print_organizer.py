import math

import numpy as np

from compas_slicer.print_organization import interpolation_print_organizer as ipo


def test_batch_closest_points_distances_are_3d():
    """Layer heights come from these distances: stacked layers must not measure as touching."""
    square = [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0], [0, 0, 0]]
    queries = [[50, 1, 10], [100, 50, 10], [50, 99, 10], [1, 50, 10]] * 3  # > 10 points: CGAL path if available
    closest, distances = ipo._batch_closest_points_on_polyline(queries, square)
    expected = [math.dist(q, c) for q, c in zip(queries, closest)]
    assert np.allclose(distances, expected)
    assert np.allclose(distances[:4], [math.sqrt(101), 10.0, math.sqrt(101), math.sqrt(101)])
