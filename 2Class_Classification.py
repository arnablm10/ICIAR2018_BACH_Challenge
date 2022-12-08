
train_cnt = []
train_cnt_sum = []
ans = 0
print
for idx, patches in enumerate(train_prob):
  count = [0] * 4
  for patch in patches:
    id = np.argmax(patch)
    if(np.std(patch) >= 0):
      count[id] += 1
  lab = np.argmax(count)
  
  train_cnt_sum.append([count[0] + count[1], count[2] + count[3]])
  train_cnt.append(count)
    
    
train_cnt = np.asarray(train_cnt, dtype = float)
train_cnt_sum = np.asarray(train_cnt_sum, dtype = float)    

test_cnt = []
test_cnt_sum = []
ans = 0
tot = 80

for idx, patches in enumerate(valid_prob):
  count = [0] * 4

  for idx2, patch in enumerate(patches):  
    id = np.argmax(patch)

    count[id] += 1
  test_cnt_sum.append([count[0] + count[1], count[2] + count[3]])  
  test_cnt.append(count)
  
  
  def two_class_classification_model(input)
    input = keras.layers.Dense(16, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer='l2')(input)
    output = keras.layers.Dense(4, activation = 'sigmoid', kernel_initializer='he_normal', kernel_regularizer='l2')(input)
    return output
