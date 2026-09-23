import rasterio
from rasterio.features import geometry_mask
import numpy as np
from scipy.ndimage import uniform_filter
import geopandas as gpd
import matplotlib.pyplot as plt
from skimage.restoration import denoise_bilateral


def normalize_image(img):
    p2, p98 = np.percentile(img,  (2, 98))
    img_stretched = np.clip(img, p2, p98)
    return (img_stretched - p2) / (p98 - p2)


def show_green_nir(green, nir, green_filt, nir_filt):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)

    axs[0, 0].imshow(green, cmap="gray")
    axs[0, 0].set_title("Green Band")
    axs[0, 1].imshow(nir, cmap="gray")
    axs[0, 1].set_title("NIR Band")
    axs[1, 0].imshow(green_filt, cmap="gray")
    axs[1, 0].set_title("Green Band (filtered)")
    axs[1, 1].imshow(nir_filt, cmap="gray")
    axs[1, 1].set_title("NIR Band (filtered)")
    
    plt.tight_layout()
    plt.show()


def crop_to_aoi(image, aoi_proj, transform):
    mask = geometry_mask(
        [geom for geom in aoi_proj.geometry],
        transform=transform,
        invert=True,  # invert=True means mask is True *inside* polygon
        out_shape=image.shape,
    )
    masked_image = np.where(mask, image, 0)
    rows = np.any(masked_image != 0, axis=1)
    cols = np.any(masked_image != 0, axis=0)

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    return masked_image[row_min:row_max+1, col_min:col_max+1]


def main(green_path, nir_path, aoi_path, denoising_methid):
    gdf = gpd.read_file(aoi_path)
    aoi_gdf = gpd.GeoDataFrame(geometry=[gdf.geometry[0]], crs=gdf.crs)
    with rasterio.open(green_path) as gsrc, rasterio.open(nir_path) as nsrc:
        aoi_proj = aoi_gdf.to_crs(gsrc.crs)
        green = gsrc.read(1).astype(np.float32)
        nir = nsrc.read(1).astype(np.float32)
        green_crop = crop_to_aoi(green, aoi_proj, gsrc.transform)
        nir_crop = crop_to_aoi(nir, aoi_proj, nsrc.transform)

    green_crop_norm = normalize_image(green_crop)
    nir_crop_norm = normalize_image(nir_crop)

    if denoising_methid == "boxcar_4x4":
        green_crop_norm_filter = uniform_filter(green_crop_norm, size=4, mode="reflect")
        nir_crop_norm_filter = uniform_filter(nir_crop_norm, size=4, mode="reflect")
    elif denoising_methid == "bilateral":
        green_crop_norm_filter = denoise_bilateral(
            green_crop_norm, sigma_color=0.1, sigma_spatial=2, channel_axis=None
        )
        nir_crop_norm_filter = denoise_bilateral(
            nir_crop_norm, sigma_color=0.1, sigma_spatial=2, channel_axis=None
        )

    show_green_nir(green_crop_norm, nir_crop_norm, green_crop_norm_filter, nir_crop_norm_filter)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Denoising examples for S2 data")
    parser.add_argument(
        "-g",
        "--green",
        type=str,
        required=True,
        help="Path to the location where the green band is stored as geotiff file",
    )
    parser.add_argument(
        "-n",
        "--nir",
        type=str,
        required=True,
        help="Path to the location where the nir band is stored as geotiff file",
    )
    parser.add_argument(
        "-a",
        "--aoi-path",
        type=str,
        required=True,
        help="Path to the location where the aoi geojson is located"
    )
    parser.add_argument(
        "-d",
        "--denoising-method",
        type=str,
        choices=["none", "boxcar_4x4", "bilateral"],
        required=True,
    )
    args = parser.parse_args()

    main(args.green, args.nir, args.aoi_path, args.denoising_method)
