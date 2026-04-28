# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.

library(keras3)

c(c(x_train, y_train), c(x_test, y_test)) %<-% keras3::dataset_mnist()
head(x_train)

str(x_train)
# int [1:60000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
dim(x_train)
# 60000    28    28

X784_train <- array_reshape(x_train, c(nrow(x_train), 784))

dim(X784_train)
#> [1] 60000   784


# ------------------------------
# Reference:
# Deep Learning Using R with keras (CNN)
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html


mnist <- dataset_mnist()
str(mnist)

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

str(x_train)
str(y_train)


index_image = 28 ## change this index to see different image.
input_matrix <- x_train[index_image,1:28,1:28]
output_matrix <- apply(input_matrix, 2, rev)
output_matrix <- t(output_matrix)
image(1:28, 1:28, output_matrix, col=gray.colors(256), xlab=paste('Image for digit of: ', y_train[index_image]), ylab="")


# Load the mnist data's training and testing dataset
# mnist <- dataset_mnist()
# x_train <- mnist$train$x
# y_train <- mnist$train$y
# x_test <- mnist$test$x
# y_test <- mnist$test$y

# Define a few parameters to be used in the CNN model
batch_size <- 128
num_classes <- 10
epochs <- 10

# Input image dimensions
img_rows <- 28
img_cols <- 28

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

str(x_train)
str(x_test)

x_train <- x_train / 255
x_test <- x_test / 255

str(x_train)
str(x_test)

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# define model structure 
cnn_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

summary(cnn_model)

# Compile model
cnn_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
cnn_history <- cnn_model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

plot(cnn_history)

cnn_model %>% evaluate(x_test, y_test)

cnn_pred <- cnn_model %>% 
  predict_classes(x_test)
head(cnn_pred, n=50)

# -------------------------------------------------------------

rotate = function(x){ t(apply(x, 2, rev)) }

imgPlot = function(img, title = ""){
  col = grey.colors(255)
  image(rotate(img), col = col, xlab = "", ylab = "", axes = FALSE,
        main = paste0("Label: ", as.character(title)))
}

data = dataset_mnist()
train = data$train
test = data$test

# Let’s visualize a few digits:
  
oldpar = par(mfrow = c(1, 3))
.n = sapply(1:3, function(x) imgPlot(train$x[x,,], train$y[x]))

par(oldpar)

train_x = array(train$x/255, c(dim(train$x), 1))
str(train_x)
test_x = array(test$x/255, c(dim(test$x), 1))
train_y = keras3::to_categorical(train$y)
test_y = keras3::to_categorical(test$y)
dim(test_y)
head(test_y)
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
# [1,]    0    0    0    0    0    0    0    1    0     0
# [2,]    0    0    1    0    0    0    0    0    0     0
# [3,]    0    1    0    0    0    0    0    0    0     0
# [4,]    1    0    0    0    0    0    0    0    0     0
# [5,]    0    0    0    0    1    0    0    0    0     0
# [6,]    0    1    0    0    0    0    0    0    0     0

model = keras_model_sequential(shape(28L, 28L, 1L))
model |>
  layer_conv_2d(filters = 16L,
                kernel_size = c(2L, 2L), activation = "relu") |>
  layer_max_pooling_2d() |>
  layer_conv_2d(filters = 16L, kernel_size = c(3L, 3L), 
                activation = "relu") |>
  layer_max_pooling_2d() |>
  layer_flatten() |>
  layer_dense(100L, activation = "relu") |>
  layer_dense(10L, activation = "softmax")

summary(model) 
# plot(model)

model |>
  keras3::compile(
    optimizer = keras3::optimizer_adamax(0.01),
    loss = keras3::loss_categorical_crossentropy,
    metrics = c('accuracy')
  )
summary(model)

library(tensorflow)
library(keras3)

epochs = 5L
batch_size = 32L

model |>
  fit(
    x = train_x, 
    y = train_y,
    epochs = epochs,
    batch_size = batch_size,
    shuffle = TRUE,
    validation_split = 0.2
  )

# -------------------------

model = keras_model_sequential(shape(28L, 28L, 1L))
model |>
  layer_conv_2d(filters = 16L,
                kernel_size = c(2L, 2L), activation = "relu") |>
  layer_max_pooling_2d() |>
  layer_conv_2d(filters = 16L, kernel_size = c(3L, 3L), 
                activation = "relu") |>
  layer_max_pooling_2d() |>
  layer_flatten() |>
  layer_dense(100L, activation = "relu") |>
  layer_dense(39L, activation = "softmax")

summary(model) 
# plot(model)

model |>
  keras3::compile(
    optimizer = keras3::optimizer_adamax(0.01),
    loss = keras3::loss_categorical_crossentropy,
    metrics = c('accuracy')
  )
summary(model)

cnn.train_history <- model |> 
  fit(x_cnn.train, 
      y_train.cat,
      epochs = epochs,
      batch_size = batch_size,
      validation_split = vld_split
  )



# model |>
#   fit(
#     x = x_cnn.train, 
#     y = y_train.cat,
#     epochs = epochs,
#     batch_size = batch_size,
#     shuffle = TRUE,
#     validation_split = 0.2
#   )


### Advanced CNN Model ---------------------------------------------------------
# Reference:
# TensorFlow 2 quickstart for experts
# https://tensorflow.rstudio.com/tutorials/quickstart/advanced

library(tensorflow)
library(tfdatasets)
library(keras3)

# Load and prepare the MNIST dataset.

c(c(x_train, y_train), c(x_test, y_test)) %<-% keras3::dataset_mnist()
x_train %<>% { . / 255 }
x_test  %<>% { . / 255 }

str(x_train)
dim(x_train)
dim(x_test)


# Use TensorFlow Datasets to batch and shuffle the dataset:
  
train_ds <- list(x_train, y_train) %>%
  tensor_slices_dataset() %>%
  dataset_shuffle(10000) %>%
  dataset_batch(32)

test_ds <- list(x_test, y_test) %>%
  tensor_slices_dataset() %>%
  dataset_batch(32)

str(train_ds)
str(test_ds)

#### Model Class
# Build the a model using the Keras model subclassing API:

my_model <- new_model_class(
  classname = "MyModel",
  initialize = function(...) {
    super$initialize()
    self$conv1 <- layer_conv_2d(filters = 32, kernel_size = 3,
                                activation = 'relu')
    self$flatten <- layer_flatten()
    self$d1 <- layer_dense(units = 128, activation = 'relu')
    self$d2 <- layer_dense(units = 10)
  },
  call = function(inputs) {
    inputs |>
      tf$expand_dims(3L) |>
      self$conv1() |>
      self$flatten() |>
      self$d1() |>
      self$d2()
  }
)

# Create an instance of the model
model <- my_model()

# Choose an optimizer and loss function for training:
loss_object <- loss_sparse_categorical_crossentropy(from_logits = TRUE)
optimizer <- optimizer_adam()

#> Select metrics to measure the loss and the accuracy of the model. 
#> These metrics accumulate the values over epochs and then print the overall result.

train_loss <- metric_mean(name = "train_loss")
str(train_loss)
train_accuracy <- metric_sparse_categorical_accuracy(name = "train_accuracy")
str(train_accuracy)

test_loss <- metric_mean(name = "test_loss")
str(test_loss)
test_accuracy <- metric_sparse_categorical_accuracy(name = "test_accuracy")
str(test_accuracy)

# Use tf$GradientTape() to train the model:

train_step <- function(images, labels) {
  with(tf$GradientTape() %as% tape, {
    # training = TRUE is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions <- model(images, training = TRUE)
    loss <- loss_object(labels, predictions)
  })
  gradients <- tape$gradient(loss, model$trainable_variables)
  optimizer$apply_gradients(zip_lists(gradients, model$trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)
}

train <- tf_function(function(train_ds) {
  for (batch in train_ds) {
    print("Hello!")
    # str(batch)
    # c(images, labels) %<-% batch
    # train_step(images, labels)
  }
})

# Test the model:

test_step <- function(images, labels) {
  # training = FALSE is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions <- model(images, training = FALSE)
  t_loss <- loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)
}

test <- tf_function(function(test_ds) {
  for (batch in test_ds) {
    c(images, labels) %<-% batch
    test_step(images, labels)
  }
})

reset_metrics <- function() {
  for (metric in list(train_loss, train_accuracy,
                      test_loss, test_accuracy))
    metric$reset_state()
}

#### Proceed Classification ----------------------------------------------------

EPOCHS <- 1
for (epoch in seq_len(EPOCHS)) {
  # Reset the metrics at the start of the next epoch
  reset_metrics()
  train(train_ds)
  test(test_ds)
  cat(sprintf('Epoch %d', epoch), "\n")
  cat(sprintf('Loss: %f', train_loss$result()), "\n")
  cat(sprintf('Accuracy: %f', train_accuracy$result() * 100), "\n")
  cat(sprintf('Test Loss: %f', test_loss$result()), "\n")
  cat(sprintf('Test Accuracy: %f', test_accuracy$result() * 100), "\n")
}


















