# TensorFlow 2 quickstart for experts ------------------------------------------
# Reference: https://tensorflow.rstudio.com/tutorials/quickstart/advanced
options(timeout = max(300, getOption("timeout")))

if(!require(tfdatasets))
  install.packages("tfdatasets")

# You may also need to run:
# tensorflow::install_tensorflow()

library(tensorflow)
library(tfdatasets)
library(keras3)
library(magrittr)

# Load and prepare the MNIST dataset.

c(c(x_train, y_train), c(x_test, y_test)) %<-% keras3::dataset_mnist()
head(x_train)
str(x_train)

x_train %<>% { . / 255 }
x_test  %<>% { . / 255 }

head(x_train)
str(x_train)
str(x_test)

# Use TensorFlow Datasets to batch and shuffle the dataset:

train_ds <- list(x_train, y_train) %>%
  tensor_slices_dataset() %>%
  dataset_shuffle(10000) %>%
  dataset_batch(32)

str(train_ds)

test_ds <- list(x_test, y_test) %>%
  tensor_slices_dataset() %>%
  dataset_batch(32)

str(test_ds)

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
    inputs %>%
      tf$expand_dims(3L) %>%
      self$conv1() %>%
      self$flatten() %>%
      self$d1() %>%
      self$d2()
  }
)

# Create an instance of the model
model <- my_model()

# Choose an optimizer and loss function for training:

loss_object <- loss_sparse_categorical_crossentropy(from_logits = TRUE)
optimizer <- optimizer_adam()













