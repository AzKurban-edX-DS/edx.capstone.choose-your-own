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



















