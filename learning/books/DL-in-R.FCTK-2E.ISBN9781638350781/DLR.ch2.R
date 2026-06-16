# Chapter 2: The Mathematical Building Blocks of Neural Network ----------------

# Listing 2.1 Loading the MNIST dataset in Keras3

library(tensorflow)
library(keras3)

mnist <- dataset_mnist()
str(mnist)

train_images <- mnist$train$x
str(train_images)

train_labels <- mnist$train$y
str(train_labels)

test_images <- mnist$test$x
str(test_images)

test_labels <- mnist$test$y
str(test_labels)


# Listing 2.2 The network architecture

model <- keras_model_sequential(layers = list(
  layer_dense(units = 512, activation = "relu"),
  layer_dense(units = 10, activation = "softmax")
))

str(model)

# Listing 2.3 The compilation step

compile(model,
        optimizer = "rmsprop",
        loss = "sparse_categorical_crossentropy",
        metrics = "accuracy")

# Listing 2.4 Preparing the image data

train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
str(train_images)

test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255
str(test_images)

# Listing 2.5 “Fitting” the model
fit(model, train_images, train_labels, epochs = 5, batch_size = 128)

# Listing 2.6 Using the model to make predictions

test_digits <- test_images[1:10, ]
str(test_digits)

predictions <- predict(model, test_digits)
str(predictions)

predictions[1, ]

which.max(predictions[1, ])
#> [1] 8

predictions[1, 8]
#> 0.9994962

#> This first test digit has the highest probability score (0.99949623, almost 1) at index 8, 
#> so according to our model, it must be a 7 (because we start counting at 0):
test_labels[1]
#> [1] 7

# Listing 2.7 Evaluating the model on new data

metrics <- evaluate(model, test_images, test_labels)
metrics["accuracy"]
# $accuracy
# [1] 0.9813

## The Engine of Neural Networks: Gradient-Based Optimization ------------------

### Chaining Derivatives: The Backpropagation Algorithm ------------------------

library(tensorflow)

x <- tf$Variable(0)

with(tf$GradientTape() %as% tape, {
  y <- 2*x + 3
})

grad_of_y_wrt_x <- tape$gradient(y,x)

# The `GradieantTape()` works with tensor operations as follows:

x <- tf$Variable(array(0, dim = c(2,2)))

with(tf$GradientTape() %as% tape, {
  y <- 2*x + 3
}) 

grad_of_y_wrt_x <- tape$gradient(y,x)

W <- tf$Variable(random_array(c(2, 2)))
b <- tf$Variable(array(0, dim = c(2)))


x <- random_array(c(2, 2))
with(tf$GradientTape() %as% tape, {
  y <- tf$matmul(x, W) + b
})

grad_of_y_wrt_W_and_b <- tape$gradient(y, list(W, b))
str(grad_of_y_wrt_W_and_b)

## 2.5 Looking Back at Our First Example ---------------------------------------

### 2.5.1 Reimplementing our First Example from scratch in `TensorFlow` --------

#### A SIMPLE DENSE CLASS ------------------------------------------------------

output <- activation(dot(W, input) + b)

layer_naive_dense <- function(input_size, output_size, activation) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveDense"
  
  
  self$activation <- activation
  
  
  w_shape <- c(input_size, output_size)
  w_initial_value <- random_array(w_shape, min = 0, max = 1e-1)
  self$W <- tf$Variable(w_initial_value)
  
  
  b_shape <- c(output_size)
  b_initial_value <- array(0, b_shape)
  self$b <- tf$Variable(b_initial_value)
  
  
  self$weights <- list(self$W, self$b)
  
  
  self$call <- function(inputs) {
    self$activation(tf$matmul(inputs, self$W) + self$b)
  }
  
  
  self
}

#### A SIMPLE SEQUENTIAL CLASS -------------------------------------------------

naive_model_sequential <- function(layers) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "NaiveSequential"
  
  
  self$layers <- layers
  
  
  weights <- lapply(layers, function(layer) layer$weights)
  self$weights <- do.call(c, weights) # ➊
  
  
  self$call <- function(inputs) {
    x <- inputs
    for (layer in self$layers)
      x <- layer$call(x)
    x
  }
  
  
  self
}

# ➊ Flatten the nested list one level.


#> Using this `NaiveDense` class and this `NaiveSequential` class, we can create a mock `Keras` model:


model <- naive_model_sequential(list(
  layer_naive_dense(input_size = 28 * 28, output_size = 512,
                    activation = tf$nn$relu),
  layer_naive_dense(input_size = 512, output_size = 10,
                    activation = tf$nn$softmax)
))
stopifnot(length(model$weights) == 4)

#### A BATCH OPERATOR ----------------------------------------------------------

#> Next, we need a way to iterate over the MNIST data in mini-batches. This is easy:


new_batch_generator <- function(images, labels, batch_size = 128) {
  self <- new.env(parent = emptyenv())
  attr(self, "class") <- "BatchGenerator"
  
  
  stopifnot(nrow(images) == nrow(labels))
  self$index <- 1
  self$images <- images
  self$labels <- labels 
  self$batch_size <- batch_size
  self$num_batches <- ceiling(nrow(images) / batch_size)
  
  
  self$get_next_batch <- function() {
    start <- self$index
    if(start > nrow(images))
      return(NULL) # ➊
    
    
    end <- start + self$batch_size - 1
    if(end > nrow(images))
      end <- nrow(images) # ➋
    
    
    self$index <- end + 1
    indices <- start:end
    list(images = self$images[indices, ],
         labels = self$labels[indices])
  }
  
  
  self
}


# ➊ Generator is finished.
# ➋ Last batch may be smaller.

one_training_step <- function(model, images_batch, labels_batch) {
  
  with(tf$GradientTape() %as% tape, {
    
    predictions <- model$call(images_batch) # ➊
    
    per_sample_losses <
      
      loss_sparse_categorical_crossentropy(labels_batch, predictions)
    
    average_loss <- mean(per_sample_losses)
    
  })
  
  gradients <- tape$gradient(average_loss, model$weights) # ➋
  
  update_weights(gradients, model$weights) # ➌
  
  average_loss
  
}

# ➊ Run the forward pass (compute the model's predictions under a GradientTape scope).

#> ➋ Compute the gradient of the loss with regard to the weights. 
#> he output gradients is a list where each entry corresponds to a weight from the model$weights list.

# ➌ Update the weights using the gradients (we will define this function shortly).

#> The simplest way to implement this `update_weights()` function is to subtract 
#> gradient * `learning_rate` from each weight:


learning_rate <- 1e-3


update_weights <- function(gradients, weights) {
  stopifnot(length(gradients) == length(weights))
  for (i in seq_along(weights))
    weights[[i]]$assign_sub( # ➊
                            gradients[[i]] * learning_rate)
}

# ➊ x$assign_sub(value) is the equivalent of x <- x - value for TensorFlow variables.

#> In practice, you would almost never implement a weight update step like this by hand.
#> Instead, you would use an Optimizer instance from Keras:


optimizer <- optimizer_sgd(learning_rate = 1e-3)


update_weights <- function(gradients, weights)
  optimizer$apply_gradients(zip_lists(gradients, weights))


#> `zip_lists()` is a helper function that we use to turn the lists of gradients 
#> and weights into a list of (gradient, weight) pairs. 
#> We use it to pair gradients with weights for the optimizer. For example:
  
  
  str(zip_lists(
    gradients = list("grad_for_wt_1", "grad_for_wt_2", "grad_for_wt_3"),
    weights = list("weight_1", "weight_2", "weight_3")))


# List of 3
# $ :List of 2
# ..$ gradients: chr "grad_for_wt_1"
# ..$ weights : chr "weight_1"
# $ :List of 2
# ..$ gradients: chr "grad_for_wt_2"
# ..$ weights : chr "weight_2"
# $ :List of 2
# ..$ gradients: chr "grad_for_wt_3"
# ..$ weights : chr "weight_3"
 
#> Now that our per-batch training step is ready, 
#> we can move on to implementing an entire epoch of training.

### 2.5.3 The full training loop -----------------------------------------------

#> An epoch of training simply consists of repeating the training step 
#> for each batch in the training data, and the full training loop 
#> is simply the repetition of one epoch:  

  fit <- function(model, images, labels, epochs, batch_size = 128) {
    for (epoch_counter in seq_len(epochs)) {
      cat("Epoch ", epoch_counter, "\n")
      batch_generator <- new_batch_generator(images, labels)
      for (batch_counter in seq_len(batch_generator$num_batches)) {
        batch <- batch_generator$get_next_batch()
        loss <- one_training_step(model, batch$images, batch$labels)
        if (batch_counter %% 100 == 0)
          cat(sprintf("loss at batch %s: %.2f\n", batch_counter, loss))
      }
    }
  }

# Let’s test-drive it:
  
  
  mnist <- dataset_mnist()
  train_images <- array_reshape(mnist$train$x, c(60000, 28 * 28)) / 255
  test_images <- array_reshape(mnist$test$x, c(10000, 28 * 28)) / 255
  test_labels <- mnist$test$y
  train_labels <- mnist$train$y
  
  
  fit(model, train_images, train_labels, epochs = 10, batch_size = 128)

  # Epoch 1
  # 
  # loss at batch 100: 2.37
  # 
  # loss at batch 200: 2.21
  # 
  # loss at batch 300: 2.15
  # 
  # loss at batch 400: 2.09
  # 
  # Epoch 2
  # 
  # loss at batch 100: 1.98
  # 
  # loss at batch 200: 1.83
  # 
  # loss at batch 300: 1.83
  # 
  # loss at batch 400: 1.75
  # 
  # 
  # …
  # 
  # 
  # Epoch 9
  # 
  # loss at batch 100: 0.85
  # 
  # loss at batch 200: 0.68
  # 
  # loss at batch 300: 0.83
  # 
  # loss at batch 400: 0.76
  # 
  # Epoch 10
  # 
  # loss at batch 100: 0.80
  # 
  # loss at batch 200: 0.63
  # 
  # loss at batch 300: 0.78
  # 
  # loss at batch 400: 0.72

### 2.5.4 Evaluating the model -------------------------------------------------

#> We can evaluate the model by taking the max.col() of its predictions over the test images, 
#> and comparing it to the expected labels:
  
  
  predictions <- model$call(test_images)
  predictions <- as.array(predictions)          # ➊
  predicted_labels <- max.col(predictions) - 1  #➋ ➌
  
  
  matches <- predicted_labels == test_labels
  cat(sprintf("accuracy: %.2f\n", mean(matches)))
  
  
  accuracy: 0.82
  
  
#  ➊ Convert the TensorFlow Tensor to an R array.
#  ➋ max.col(x) is a vectorized implementation of apply(x, 1, which.max)).
#> ➌ Subtract 1 because positions are offset from labels by 1, for example, 
#> the first position corresponds to digit 0.

  
  
  
  
  
  
  
  
  
  
  
  






