from perlin_numpy import generate_perlin_noise_3d
import matplotlib.pyplot as plt

def generate_tiled_noise_3d(shape=(256, 256, 256), res=(8,8,8)):
    # shape must be a multiple of res
    # tileable=(Z, Y, X) - set to True for the dimensions you want to repeat
    noise = generate_perlin_noise_3d(
        shape=shape, 
        res=res, 
        tileable=(True, True, True)
    )

    mn = noise.min()
    mx = noise.max()
    noise = (noise - mn) / (mx - mn)
    noise = noise ** 2

    print(noise.shape, noise.size)
    print(noise.max())
    print(noise.min())

    with open('data/disk3d.txt', 'w') as buff:
        buff.write(f'{noise.shape[0]} {noise.shape[1]} {noise.shape[2]}\n')
        for v in noise.reshape(-1):
            buff.write(f'{v}\n')

    slc = noise[noise.shape[0] // 2]
    plt.imshow(slc, cmap='gray')
    plt.show()

if __name__ == '__main__':
    generate_tiled_noise_3d()
