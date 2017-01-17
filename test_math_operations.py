import numpy as np
from math_operations import *

def test_math1():
  mat = np.arange(16).reshape((4, 4))
  assert (math1(mat) == np.array([[  0,   1,   4,   9],
       [ 16,  25,  36,  49],
       [ 64,  81, 100, 121],
       [144, 169, 196, 225]])).all()
  assert (math1(mat.T) == np.array([[  0,  16,  64, 144],
       [  1,  25,  81, 169],
       [  4,  36, 100, 196],
       [  9,  49, 121, 225]])).all()


def test_math2():
  mat = np.arange(16).reshape((4, 4))
  assert (math2(mat) == 120)
  assert (math2(mat.T) == 120)


def test_math3():
  mat = np.arange(16).reshape((4, 4))
  assert (math3(mat) == np.array([24, 28, 32, 36])).all()
  assert (math3(mat.T) == np.array([ 6, 22, 38, 54])).all()