import numpy as np
from matplotlib import pyplot as plt
import time

def main():
  img = plt.imread('lena.png')#.astype(np.float)/255
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


def convolution(image, kernel):
  """
  Write a general function to convolve an image with an arbitrary kernel.
  """
  out = np.zeros(image.shape)
  kernel = kernel[::-1, ::-1] #Flipping kernel to follow convention
  N, M, C = image.shape
  Nk, Mk = kernel.shape
  nk_2 = Nk // 2
  mk_2 = Mk // 2
  for i in range(nk_2, N - nk_2):
    for j in range(mk_2, M - mk_2):
      for c in range(C):
        out[i, j, c] = np.sum((image[i-nk_2:i+nk_2+1, j-nk_2:j+nk_2+1, c]*kernel))
  return out


def blur_filter(img):
  """
  Use your convolution function to filter your image with an average filter (box filter)
  with kernal size of 11.
  """
  k_size = 11
  kernel = np.ones((k_size, k_size))/k_size**2
  return convolution(img, kernel)


def sobel_filter(img):
  """
  Use your convolution function to filter your image with a sobel operator
  """
  kernel = [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
  kernel = np.array(kernel)
  return convolution(img, kernel)




if __name__ == '__main__':
  main()