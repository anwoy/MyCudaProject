import numpy as np
import imageio
import openexr_numpy as exr
import matplotlib.pyplot as plt


def make_starmap_png(swap_bgr_to_rgb=False, gamma_correction=np.array([1., 1., 1.])):
    '''
    Preprocesses the exr image and converts it to png
    
    :param swap_bgr_to_rgb: if image is in BGR format instead of RGB, then set this as `True`
    :param gamma_correction: apply gamma correction to individual channels
    '''
    file_path = "data/starmap_2020_4k.exr"

    # Load the data
    image_data = exr.imread(file_path).astype(np.float32)

    # Swap BGR to RGB
    if swap_bgr_to_rgb:
        image_data_rgb = image_data[..., ::-1]
    else:
        image_data_rgb = image_data

    # clipping and gamma correction
    processed = np.clip(image_data_rgb, 0, 1)
    processed = np.power(processed, 1/gamma_correction)

    # Convert to 8-bit Integer (0-255)
    final_png = (processed * 255).astype(np.uint8)
    print(
        f'final_png shape = {final_png.shape}, max_pixel_val = {final_png.max()}, min_pixel_val = {final_png.min()}')

    # Save the file
    output_path = "data/starmap.png"
    imageio.imwrite(output_path, final_png)

    # Visualize to verify colors
    plt.imshow(final_png)
    plt.show()


if __name__ == "__main__":
    make_starmap_png()
