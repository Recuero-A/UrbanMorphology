import numpy as np
from shapely.geometry import box

from urbanmorph.preprocessing.projection import SquareBounds
from urbanmorph.preprocessing.rasterize import pixel_centers, rasterize_geometries

B = SquareBounds(0.0, 0.0, 16.0, 16.0)  # 4x4 grid of 4 m pixels


def test_pixel_centres_orientation():
    X, Y = pixel_centers(B, 4)
    assert X[0, 0] == 2 and Y[0, 0] == 14  # row 0 is the NORTH edge
    assert X[0, 3] == 14 and Y[3, 0] == 2


def test_square_fills_expected_pixels():
    mask = rasterize_geometries([box(0, 8, 8, 16)], B, 4)  # north-west quarter
    expected = np.zeros((4, 4), np.uint8)
    expected[:2, :2] = 1
    np.testing.assert_array_equal(mask, expected)
    assert mask.dtype == np.uint8


def test_centre_rule_drops_subpixel_features():
    # A 1 m wide strip that avoids every pixel centre disappears entirely.
    assert rasterize_geometries([box(3.0, 0, 4.0, 16)], B, 4).sum() == 0


def test_boundary_counts_as_inside():
    # Polygon edge exactly through a pixel-centre column x=2.
    assert rasterize_geometries([box(2, 0, 3, 16)], B, 4)[:, 0].sum() == 4


def test_empty_input_gives_zeros():
    assert rasterize_geometries([], B, 4).sum() == 0
