def custom_training(X, Y, XV, YV, BATCH_SIZE, NUM_OF_EPOCHS, STEPS_PER_EPOCH, model):
  optimizer = tf.keras.optimizers.SGD()
  loss_fn = tf.keras.losses.CategoricalCrossentropy()

  train_loss_result = []
  train_accuracy_result = []

  for epoch in range(NUM_OF_EPOCHS): 
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    valid_accuracy = tf.keras.metrics.CategoricalAccuracy()

    # temp = list(zip(X, Y))
    # random.shuffle(temp)
    # X, Y = zip(*temp)
  
    X, Y = shuffle(X, Y, random_state = 0)
    # print(X.shape)
    idx = 0
    cnt = 0
    
    while cnt < STEPS_PER_EPOCH:
      x_batch = X[idx : idx + BATCH_SIZE]
      y_batch = Y[idx : idx + BATCH_SIZE]

      
      with tf.GradientTape() as tape:
        # print(idx)
        y_pred = model(x_batch, training=False)
        loss = loss_fn(y_batch, y_pred)

      # if epoch % 20 == 0 and epoch > 0:
      #   for i in range(BATCH_SIZE):
      #     tmp = [0] * 4
      #     id = np.argmax(y_pred[i])
      #     tmp[id] = 1
      #     tmp = np.array(tmp)
      #     Y[idx + i] = tmp  
      idx += BATCH_SIZE
      cnt += 1

      epoch_loss_avg.update_state(loss)
      grad = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grad, model.trainable_variables))
      epoch_accuracy.update_state(y_batch, y_pred)


    train_accuracy_result.append(epoch_accuracy.result().numpy())
    train_loss_result.append(epoch_loss_avg.result().numpy())
    print("Loss: ", epoch_loss_avg.result().numpy(), " Accuracy : ", epoch_accuracy.result().numpy())
    y_pred = model(XV, training = False)

    valid_accuracy.update_state(YV, y_pred)
    valid_loss = loss_fn(YV, y_pred)
    print("VALID loss: ", valid_loss.numpy(), "  VALID Accuracy : ", valid_accuracy.result().numpy())
    print(epoch, " epoch done")
