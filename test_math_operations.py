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


def test_math4():
  mat = np.arange(16).reshape((4, 4))
  assert math4(mat).sum() == 1
  assert math4(mat.T).sum() == 1


def test_math5():
  mat = np.arange(16).reshape((4, 4))
  vec = np.arange(4)
  assert (math5(mat, vec) == np.array([[ 0,  0,  0,  0],
       [ 4,  5,  6,  7],
       [16, 18, 20, 22],
       [36, 39, 42, 45]])).all()
  assert (math5(mat.T) == np.array([[ 0,  0,  0,  0],
       [ 1,  5,  9, 13],
       [ 4, 12, 20, 28],
       [ 9, 21, 33, 45]])).all()


def test_math6():
  mat = np.arange(16).reshape((4, 4))
  assert (math6(mat) == np.array([24, 28, 32, 36])).all()
  assert (math6(mat.T) == np.array([ 6, 22, 38, 54])).all()


def test_math7():
  mat = np.arange(16).reshape((4, 4))
  assert (math7(mat) == np.array([24, 28, 32, 36])).all()
  assert (math7(mat.T) == np.array([ 6, 22, 38, 54])).all()