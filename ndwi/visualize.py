from pathlib import Path

import matplotlib.pyplot as plt
import tifffile
import numpy as np
from ndwi import calculate_ndwi


DATA_DIR = Path(__file__).parent

green = tifffile.imread(
    DATA_DIR / "S2C_31TDF_20250630_0_L2A_green.tif"
)

nir = tifffile.imread(
    DATA_DIR / "S2C_31TDF_20250630_0_L2A_nir.tif"
)

ndwi = calculate_ndwi(green, nir)

print("Green:", green.shape, green.dtype)
print("NIR:", nir.shape, nir.dtype)
print("NDWI:", ndwi.shape, ndwi.dtype)
print("NDWI min:", np.nanmin(ndwi))
print("NDWI max:", np.nanmax(ndwi))

plt.imshow(ndwi, cmap="RdYlBu")
plt.title("NDWI")
plt.colorbar(label="NDWI")
plt.show()