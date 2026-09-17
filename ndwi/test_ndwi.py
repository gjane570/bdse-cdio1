import numpy as np
import pytest
from ndwi import calculate_ndwi


def test_ndwi_basic():
    green = np.array([3.0])
    nir = np.array([1.0])

    result = calculate_ndwi(green, nir)

    expected = np.array([0.5])

    np.testing.assert_array_equal(result, expected)

def test_ndwi_division_by_zero():
    green = np.array([0.0])
    nir = np.array([0.0])

    result = calculate_ndwi(green, nir)

    assert np.isnan(result[0])




@pytest.mark.parametrize(
    "green, nir, expected",
    [
        (3.0, 1.0, 0.5),
        (1.0, 3.0, -0.5),
        (5.0, 5.0, 0.0),
        (10.0, 0.0, 1.0),
        (0.0, 10.0, -1.0),
    ],
)
def test_ndwi_different_inputs(green, nir, expected):
    result = calculate_ndwi(
        np.array([green]),
        np.array([nir]),
    )

    np.testing.assert_allclose(result, np.array([expected]))

def test_ndwi_missing_values():
    green = np.array([3.0, np.nan])
    nir = np.array([1.0, 2.0])

    result = calculate_ndwi(green, nir)

    assert np.isclose(result[0], 0.5)
    assert np.isnan(result[1])