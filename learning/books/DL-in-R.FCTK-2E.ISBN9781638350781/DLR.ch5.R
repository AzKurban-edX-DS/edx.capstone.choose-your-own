#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Chapter 5. Fundamentals of Machine Learning
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


## 5.1 Generalization:  The Goal of Machine Learning --------------------------

#> Listing 5.7 Training an MNIST model with an incorrectly high learning rate --



c(c(train_images, train_labels), .) %<-% dataset_mnist()
train_images <- array_reshape(train_images / 255,
                              c(60000, 28 * 28))


model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")


model %>% compile(optimizer = optimizer_rmsprop(1),
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")


history <- model %>% fit(train_images, train_labels,
                         epochs = 10, batch_size = 128,
                         validation_split = 0.2)
plot(history)

#> Listing 5.8 The same model with a more appropriate learning rate ------------

model <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")


model %>% compile(optimizer = optimizer_rmsprop(1e-2),
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")


model %>%
  fit(train_images, train_labels,
      epochs = 10, batch_size = 128,
      validation_split = 0.2) ->
  history
plot(history)

#> Listing 5.9 A simple logistic regression on MNIST ---------------------------



model <- keras_model_sequential() %>%
  layer_dense(10, activation = "softmax")


model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")


history_small_model <- model %>%
  fit(train_images, train_labels,
      epochs = 20,
      batch_size = 128,
      validation_split = 0.2)


plot(history_small_model$metrics$val_loss, type = 'o',
     main = "Effect of Insufficient Model Capacity on Validation Loss",
     xlab = "Epochs", ylab = "Validation Loss")


plot(history_small_model)

## 5.4 Improving generalization ------------------------------------------------

### 5.4.4 Regularizing your model ----------------------------------------------

#### REDUCING THE NETWORK’S SIZE -----------------------------------------------

##### Listing 5.10 Original model ----

c(c(train_data, train_labels), .) %<-% dataset_imdb(num_words = 10000)

str(train_data)
str(train_labels)

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for(i in seq_along(sequences))
    results[i, sequences[[i]]] <- 1
  results
}


train_data <- vectorize_sequences(train_data)
str(train_data)

model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(16, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")


model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


history_original <- model %>%
  fit(train_data, train_labels,
      epochs = 20, batch_size = 512, validation_split = 0.4)

plot(history_original)

# Now let’s try to replace it with this smaller model.

##### Listing 5.11 Version of the model with lower capacity -----

model <- keras_model_sequential() %>%
  layer_dense(4, activation = "relu") %>%
  layer_dense(4, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")

model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


history_smaller_model <- model %>%
  fit(train_data, train_labels,
      epochs = 20, batch_size = 512, validation_split = 0.4)

plot(history_smaller_model)

#> Let’s generate a plot (figure 5.17) to compare the validation losses 
#> of the original model and the smaller model:
  
  plot(
    NULL, # ➊
    main = "Original Model vs. Smaller Model on IMDB Review Classification",
    xlab = "Epochs",
    xlim = c(1, history_original$params$epochs),
    ylab = "Validation Loss",
    ylim = extendrange(history_original$metrics$val_loss),
    panel.first = abline(v = 1:history_original$params$epochs, # ➋
                         lty = "dotted", col = "lightgrey")
  )

lines(history_original $metrics$val_loss, lty = 2)
lines(history_smaller_model$metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of smaller model"))

# ➊ NULL tells plot() to set up the plot region but not draw any data yet.
# ➋ Draw grid lines.

##### Listing 5.12 Version of the model with higher capacity ----

model <- keras_model_sequential() %>%
  layer_dense(512, activation = "relu") %>%
  layer_dense(512, activation = "relu") %>%
  layer_dense(1, activation = "sigmoid")


model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


history_larger_model <- model %>%
  fit(train_data, train_labels,
      epochs = 20, batch_size = 512, validation_split = 0.4)
plot(
  NULL,
  main =
    "Original Model vs. Much Larger Model on IMDB Review Classification",
  xlab = "Epochs", xlim = c(1, history_original$params$epochs),
  ylab = "Validation Loss",
  ylim = range(c(history_original$metrics$val_loss,
                 history_larger_model$metrics$val_loss)),
  panel.first = abline(v = 1:history_original$params$epochs,
                       lty = "dotted", col = "lightgrey")
)
lines(history_original $metrics$val_loss, lty = 2)
lines(history_larger_model$metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of larger model"))


#> Let’s try training a bigger model, one with two intermediate layers with 96 units each:

model <- keras_model_sequential() %>%
  layer_dense(96, activation = "relu") %>%
  layer_dense(96, activation = "relu") %>%
  layer_dense(10, activation = "softmax")


model %>% compile(optimizer = "rmsprop",
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")

history_large_model <- model %>%
  fit(train_images, train_labels,
      epochs = 20,
      batch_size = 128,
      validation_split = 0.2)

plot(history_large_model)

plot(history_large_model$metrics$val_loss, type = 'o',
     main = "Validation Loss for a Model with Appropriate Capacity",
     xlab = "Epochs", ylab = "Validation Loss")


##### Listing 5.13 Adding L2 weight regularization to the model ----------------

model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu",
              kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dense(16, activation = "relu",
              kernel_regularizer = regularizer_l2(0.002)) %>%
  layer_dense(1, activation = "sigmoid")


model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


history_l2_reg <- model %>% fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512, validation_split = 0.4)


plot(history_l2_reg)

##### Listing 5.14 Generating a plot to demonstrate the effect of L2 weight regularization ----

plot(NULL,
     main = "Effect of L2 Weight Regularization on Validation Loss",
     xlab = "Epochs",
     xlim = c(1, history_original$params$epochs),
     ylab = "Validation Loss",
     ylim = range(c(history_original$metrics$val_loss,
                    history_l2_reg $metrics$val_loss)),
     panel.first = abline(v = 1:history_original$params$epochs,
                          lty = "dotted", col = "lightgrey"))
lines(history_original$metrics$val_loss, lty = 2)
lines(history_l2_reg $metrics$val_loss, lty = 1)
legend("topleft", lty = 2:1,
       legend = c("Validation loss of original model",
                  "Validation loss of L2-regularized model"))

##### Listing 5.15 Different weight regularizers available in Keras

regularizer_l1(0.001) # ➊
regularizer_l1_l2(l1 = 0.001, l2 = 0.001)


# <keras.regularizers.L1 object at 0x7f81cc3df340> # ➋
# <keras.regularizers.L1L2 object at 0x7f81cc651c40>
#   
#   
# ➊ L1 regularization
# ➋ Simultaneous L1 and L2 regularization

##### Listing 5.16 Adding dropout to the IMDB model ----

model <- keras_model_sequential() %>%
  layer_dense(16, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(16, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")


model %>% compile(optimizer = "rmsprop",
                  loss = "binary_crossentropy",
                  metrics = "accuracy")


history_dropout <- model %>% fit(
  train_data, train_labels,
  epochs = 20, batch_size = 512,
  validation_split = 0.4
)


plot(history_dropout)

##### Listing 5.17 Generating a plot to demonstrate the effect of dropout on validation loss ----

plot(NULL,
     main = "Effect of Dropout on Validation Loss",
     xlab = "Epochs", xlim = c(1, history_original$params$epochs),
     ylab = "Validation Loss",
     ylim = range(c(history_original$metrics$val_loss,
                    history_dropout $metrics$val_loss)),
     
     panel.first = abline(v = 1:history_original$params$epochs,
                          lty = "dotted", col = "lightgrey"))

lines(history_original$metrics$val_loss, lty = 2)
lines(history_dropout $metrics$val_loss, lty = 1)

legend("topleft", lty = 1:2,
       legend = c("Validation loss of dropout-regularized model",
                  "Validation loss of original model"))
























