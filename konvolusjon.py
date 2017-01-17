import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from numba import jit #Bruk numba, hvis man har darlig tid
import time

def main():
  img = imread('lena.jpg').astype(np.float)/255
  plt.imshow(img)

  start = time.time()
  out1 = sobel_filter(img)
  out2 = blur_filter(img)
  print 'Calculation time:', time.time()-start, 'sec'
  plt.figure()
  plt.imshow(out1.mean(2), vmin=out1.min(), vmax=out1.max(), cmap='gray')
  plt.figure()
  plt.imshow(out2, vmin=out2.min(), vmax=out2.max())
  plt.show()


def blur_filter(img):
  k_size = 11
  kernel = np.ones((k_size, k_size))/k_size**2
  return convolution(img, kernel[::-1, ::-1])


def sobel_filter(img):
  kernel = [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
  kernel = np.array(kernel)
  return convolution(img, kernel[::-1, ::-1])


@jit(nopython=True)
def convolution(image, kernel):
  out = np.zeros(image.shape)
  #Flipping kernel to follow convention
  N, M, C = image.shape
  Nk, Mk = kernel.shape
  nk_2 = Nk // 2
  mk_2 = Mk // 2
  for i in range(nk_2, N - nk_2):
    for j in range(mk_2, M - mk_2):
      for c in range(C):
        out[i, j, c] = np.sum((image[i-nk_2:i+nk_2+1, j-nk_2:j+nk_2+1, c]*kernel))
  return out


if __name__ == '__main__':
  main()