import numpy as np
import imageio
import openexr_numpy as exr
import matplotlib.pyplot as plt

def make_starmap_png(swap_bgr_to_rgb=False, gamma_correction=np.array([1., 1., 1.])):
    file_path = "data/starmap_2020_4k.exr"
    
    # 1. Load the data (currently in BGR order due to alphabetical sorting)
    image_data = exr.imread(file_path).astype(np.float32)

    # 2. Swap BGR to RGB
    if swap_bgr_to_rgb:
        image_data_rgb = image_data[..., ::-1]
    else:
        image_data_rgb = image_data

    # 3. Tone Mapping & Gamma Correction
    # clipping at 1.0 ensures stars don't exceed PNG limits
    # power(1/2.2) corrects linear EXR data for standard monitors
    processed = np.clip(image_data_rgb, 0, 1)
    processed = np.power(processed, 1/gamma_correction)

    # 4. Convert to 8-bit Integer (0-255)
    final_png = (processed * 255).astype(np.uint8)
    print(f'final_png shape = {final_png.shape}, max_pixel_val = {final_png.max()}, min_pixel_val = {final_png.min()}')

    # 5. Save the file
    output_path = "data/starmap.png"
    imageio.imwrite(output_path, final_png)

    # 6. Visualize to verify colors
    plt.imshow(final_png)
    plt.title("Corrected RGB Starmap")
    plt.show()

if __name__ == "__main__":
    make_starmap_png()
