import numpy as np
import scipy.ndimage as ndimage

def estimate_coastline(waterbody_array):
    # Obté la linia de costa mitjançant el gradient morforlògic (dilatació - erosió) 
    
    kernel = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(waterbody_array, structure=kernel)
    eroded = ndimage.binary_erosion(waterbody_array, structure=kernel)
    
    coastline = dilated.astype(int) - eroded.astype(int)
    return coastline.astype(np.uint8)