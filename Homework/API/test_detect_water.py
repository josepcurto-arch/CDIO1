import numpy as np
from detect_water import detect_waterbody

def test_detect_waterbody():
    # Matriz NDWI de prova
    test_ndwi = np.array([
        [0.5, -0.2, 0.0],
        [np.nan, 0.1, -0.8]
    ])
    
    # Resultat esperat:
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.uint8)
    
    result = detect_waterbody(test_ndwi)
    
    # Comprova que detect_waterbody torni exactament la matriu esperada
    np.testing.assert_array_equal(result, expected)

    # ------
    # Només comprova que el codi funcioni com s'espera.