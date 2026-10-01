import pytest
import numpy as np

def compute_ndwi(green, nir):
    """
    Calcula el NDWI gestionant divisions per zero i valors nuls/invalids.
    """
    denominator = green + nir
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = np.where(np.abs(denominator) > 1e-6, (green - nir) / denominator, 0.0)
        ndwi = np.nan_to_num(ndwi, nan=0.0, posinf=1.0, neginf=-1.0)
    return ndwi

# --- TESTS UNITARIS ---

def test_ndwi_standard_case():
    # Cas estàndard amb valors normals
    green = np.array([0.4, 0.5])
    nir = np.array([0.2, 0.1])
    # (0.4 - 0.2)/(0.4 + 0.2) = 0.2/0.6 = 0.3333...
    expected = np.array([1/3, 4/6])
    result = compute_ndwi(green, nir)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_ndwi_division_by_zero():
    # Cas límit: Denominador és zero (Green = 0, NIR = 0)
    green = np.array([0.0])
    nir = np.array([0.0])
    result = compute_ndwi(green, nir)
    assert result[0] == 0.0

def test_ndwi_with_nan_and_inf():
    # Cas límit: Presència de NaN i Inf
    green = np.array([np.nan, 0.3])
    nir = np.array([0.1, np.inf])
    result = compute_ndwi(green, nir)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()