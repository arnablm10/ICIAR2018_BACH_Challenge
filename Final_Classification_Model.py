def final_classification_model(input):
  input = keras.layers.Dense(16, activation='sigmoid', kernel_regularizer='l2', kernel_initializer='he_uniform')(input)
  input = keras.layers.Dense(4, activation='sigmoid', kernel_regularizer='l2', kernel_initializer='he_uniform')(input)
  output = keras.layers.Dense(4, activation='softmax', kernel_regularizer='l2', kernel_initializer='he_uniform')(input)
  
  return output


def cal_dice_score(Y_test, y_pred):
  print(Y_train.shape)
  ##calculating true positives
  m = keras.metrics.TruePositives()
  m.update_state(Y_test, y_pred)
  tp = m.result().numpy()
  # print(tp)

  ##calculating false positives
  m = keras.metrics.FalsePositives()
  m.update_state(Y_test, y_pred)
  fp = m.result().numpy()
  # print(fp)

  ##calculating false negatives
  m = keras.metrics.FalseNegatives()
  m.update_state(Y_test, y_pred)
  fn = m.result().numpy()
  # print(fn)

  m = keras.metrics.TrueNegatives()
  m.update_state(Y_test, y_pred)
  tn = m.result().numpy()
  # print(tn)

  print("TOTAL: ", fp + fn + tp + tn)
  Dice_Score = 2 * tp / (2 * tp + fp + fn)
  
  # Other metrics to be calculated in same way
  print("DICE SCORE: ", Dice_Score)
