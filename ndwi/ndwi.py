import numpy as np


def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    green = green.astype(float)
    nir = nir.astype(float)

    denominator = green + nir

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / denominator

    ndwi[denominator == 0] = np.nan

    return ndwi