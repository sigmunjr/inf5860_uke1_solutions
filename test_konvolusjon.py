import numpy as np
from skimage.io import imread



def test_sobel_filter():
  from konvolusjon import sobel_filter
  img = imread('lena.jpg').astype(np.float) / 255
  out = sobel_filter(img)
  out -= out.min()
  out /= out.max()
  out = (255*out).astype(np.uint8)
  correct = imread('sobel_lena.png')
  assert np.abs(out - correct).sum() == 0

def test_blur_filter():
  from konvolusjon import blur_filter
  img = imread('lena.jpg').astype(np.float) / 255
  out = blur_filter(img)
  out -= out.min()
  out /= out.max()
  out = (255*out).astype(np.uint8)
  correct = imread('blur_lena.png')
  assert np.abs(out - correct).sum() == 0