## Not run: 
# library(reticulate)
# py_require(python_version = "3.11")
# py_require()
# 
# library(keras3)
library(tensorflow)
library(keras)
py_require_tensorflow()
library(autokeras)

# use the MNIST dataset as an example
mnist <- dataset_mnist()
c(x_train, y_train) %<-% mnist$train
c(x_test, y_test) %<-% mnist$test


# Initialize the image classifier
clf <- model_image_classifier(max_trials = 10) %>% # It tries 10 different models
  fit(x_train, y_train) # Feed the image classifier with training data

# Predict with the best model
(predicted_y <- clf %>% predict(x_test))

# Evaluate the best model with testing data
clf %>% evaluate(x_test, y_test)

# Get the best trained Keras model, to work with the keras R library
export_model(clf)

## End(Not run)
