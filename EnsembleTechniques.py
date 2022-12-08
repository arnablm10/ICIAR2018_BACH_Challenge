def max_ensemble(outputs):
  ret = np.amax(outputs, axis = 0)
  print(ret.shape)
  return ret
  

def product_ensemble(outputs):
  ret = np.prod(outputs, axis = 0)
  print(ret.shape)
  return ret


def average_ensemble(outputs):
  ret = np.average(outputs, axis = 0)
  print(ret.shape)
  return ret

def normal(outputs, i):
  ret = outputs[i]
  print(ret.shape)
  return ret
