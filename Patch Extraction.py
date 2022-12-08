def extract_patches(images, id):
  patches = []
  labels = []
  for idx, image in enumerate(images):
    tmp = tf.reshape(image, [1, Image_height, Image_width, 3])
    # print(tmp.shape)
    patch = tf.image.extract_patches(tmp, [1, patch_dim, patch_dim, 1], [1, stride , stride, 1], [1, 1, 1, 1], 'VALID')
    

    # print(patch_count)
    # print("hi")
    # print(patch.shape)
    patch = tf.reshape(patch, [patch_count, patch_dim, patch_dim, 3])
    patch = np.asarray(patch)
    # print(patch.shape)
    patch = list(patch)
    patches = patches + patch

    for _ in range(patch_count):
      labels.append(id[idx])

  patches = np.asarray(patches)
  # print(patches.shape) 
  labels = np.asarray(labels)
  return patches, labels 
