def extract_patch_features(X, feat_extraction_model):
  patch_features = []
  for idx, patch in enumerate(X):
    # print(patch.shape)
    patch = tf.reshape(patch, [1, patch_dim, patch_dim, 3])
    patch_features.append(feat_extraction_model(patch, training = False))

  patch_features = np.asarray(patch_features)
  print(patch_features.shape)
  patch_features = np.squeeze(patch_features, axis = 1)
  print(patch_features.shape)
  return patch_features
