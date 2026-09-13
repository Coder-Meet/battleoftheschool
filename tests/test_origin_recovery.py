import numpy as np
from scipy import ndimage as ndi

from detector import wall_origin
from origin_recovery import connected_wall_root


def bent_connector():
    parent = np.zeros((25, 25, 25), dtype=bool)
    parent[:, :, :4] = True
    support = parent.copy()
    support[12, 7, 4:8] = True
    support[12, 7:11, 7] = True
    support[12, 10, 7:14] = True
    excluded = np.zeros_like(parent)
    return parent, support, excluded


def test_curved_supported_connector_recovers_without_bridging_background():
    parent, support, excluded = bent_connector()
    path = np.array([[12, 10, 7], [12, 10, 8], [12, 10, 9]], dtype=float)
    distance = ndi.distance_transform_edt(~parent) - ndi.distance_transform_edt(parent)
    assert wall_origin(path, parent, support, distance, 1.0) is None
    relocated = connected_wall_root(path[0].astype(np.int64), parent, support, excluded, 1.0)
    np.testing.assert_array_equal(relocated, [12, 7, 4])


def test_disconnected_vessel_remains_rejected():
    parent, support, excluded = bent_connector()
    support[12, 7, 5] = False
    assert connected_wall_root(np.array([12, 10, 7]), parent, support, excluded, 1.0) is None


def test_cap_exclusion_blocks_recovery():
    parent, support, excluded = bent_connector()
    excluded[:, :, 4] = True
    assert connected_wall_root(np.array([12, 10, 7]), parent, support, excluded, 1.0) is None


def test_connector_length_is_bounded_in_physical_units():
    parent, support, excluded = bent_connector()
    root = np.array([12, 10, 8])
    assert connected_wall_root(root, parent, support, excluded, 1.0) is None
    np.testing.assert_array_equal(
        connected_wall_root(root, parent, support, excluded, 0.5), [12, 7, 4],
    )
