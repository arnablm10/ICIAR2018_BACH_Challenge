pre_trained_models = [VGG16, VGG19, InceptionResNetV2]

def ensembling_pre_trained_models(pre_trained_models):
  train_prob = []
  valid_prob = []
  test_prob = []

  for pre_trained_model in pre_trained_models:
    base_model = pre_trained_model(include_top=False, weights='imagenet', input_shape=(patch_dim, patch_dim, 3))

    bmoutput = base_model.output
    print(bmoutput.shape)
    bmoutput = keras.layers.GlobalAveragePooling2D()(bmoutput)
    feat_extraction_model = keras.Model(inputs=base_model.input, outputs=bmoutput)

    print(bmoutput.shape)
    patch_features = extract_patch_features(X_train, feat_extraction_model)
    valid_patch_features = extract_patch_features(X_valid, feat_extraction_model)
    test_patch_features = extract_patch_features(X_test, feat_extraction_model)

    model = Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(Dense(16, kernel_regularizer='l2', kernel_initializer='he_uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('sigmoid'))
    model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.BatchNormalization())
    # model.add(Dense(32, kernel_regularizer='l2', kernel_initializer='he_uniform'))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Activation('sigmoid'))


    model.add(keras.layers.Dense(4, activation = 'softmax'))
    print("Training model", pre_trained_model)

    custom_training(patch_features, y_train, valid_patch_features, y_valid, 32, 50, 261, model)
    train_prob.append(model(patch_features))
    valid_prob.append(model(valid_patch_features))
    test_prob.append(model(test_patch_features))

  train_prob = np.asarray(train_prob)
  valid_prob = np.asarray(valid_prob)
  test_prob = np.asarray(test_prob)
  return train_prob, valid_prob, test_prob
