#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Chapter 7: Working with Kearas: A deep dive
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## 7.2 Different Ways to Build Keras Models ------------------------------------

### 7.2.2 The Functional API ---------------------------------------------------

#### A SIMPLE EXAMPLE ----------------------------------------------------------

##### Listing 7.8 A simple Functional model with two Dense layers ----

# (1)
inputs <- layer_input(shape = c(3), name = "my_input")
inputs
# <KerasTensor shape=(None, 3), dtype=float32, sparse=False, ragged=False, name=my_input>

# (2)
features <- inputs %>% layer_dense(64, activation = "relu")
features
# <KerasTensor shape=(None, 64), dtype=float32, sparse=False, ragged=False, name=keras_tensor_101>

# (3)
outputs <- features %>% layer_dense(10, activation = "softmax")

# (4)
model <- keras_model(inputs = inputs, outputs = outputs)

#> Let’s go over this step by step. We started by declaring a layer_input() 
#> (note that you can also give names to these input objects, like everything else):

# (1)
inputs <- layer_input(shape = c(3), name = "my_input")
inputs

str(inputs$shape)
# List of 2
#  $ : NULL
#  $ : int 3

inputs$shape
# [[1]]
# NULL
# 
# [[2]]
# [1] 3

inputs$dtype
# [1] "float32"

# (2)
layer_instance <- layer_dense(units = 64, activation = "relu")
layer_instance
# <Dense name=dense_25, built=False>
#  signature: (*args, **kwargs)

features <- layer_instance(inputs)
features
# <KerasTensor shape=(None, 64), dtype=float32, sparse=False, ragged=False, name=keras_tensor_102>

str(features$shape)
# List of 2
#  $ : NULL
#  $ : int 64

features$shape
# [[1]]
# NULL
# 
# [[2]]
# [1] 64

dim(features)

## 7.3 Using Build-in Training and Evaluation Loops ----------------------------

#> Listing 7.17 The standard workflow: compile(), fit(), evaluate(), predict()



get_mnist_model <- function() {                               # ➊
  inputs <- layer_input(shape = c(28 * 28))
  outputs <- inputs %>%
    layer_dense(512, activation = "relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(10, activation = "softmax")
  
  
  keras_model(inputs, outputs)
}


c(c(images, labels), c(test_images, test_labels)) %<-%        # ➋
  dataset_mnist()


images <- array_reshape(images, c(-1, 28 * 28)) / 255
test_images <- array_reshape(test_images, c(-1, 28 * 28)) / 255


val_idx <- seq(10000)
val_images <- images[val_idx, ]
val_labels <- labels[val_idx]
train_images <- images[-val_idx, ]
train_labels <- labels[-val_idx]


model <- get_mnist_model()
model %>% compile(optimizer = "rmsprop", #                      ➌
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")
model %>% fit(train_images, train_labels, #                     ➍
              epochs = 3,
              validation_data = list(val_images, val_labels))
test_metrics <- model %>% evaluate(test_images, test_labels) #  ➎
predictions <- model %>% predict(test_images) #                 ➏


# ➊ Create a model (we factor this into a separate function so as to reuse it later).
# ➋ Load your data, reserving some for validation.
# ➌ Compile the model by specifying its optimizer, the loss function to minimize, and the metrics to monitor.
# ➍ Use fit() to train the model, optionally providing validation data to monitor performance on unseen data.
# ➎ Use evaluate() to compute the loss and metrics on new data.
# ➏ Use predict() to compute classification probabilities on new data.

##### Listing 7.19 Using the callbacks argument in the fit() method



callbacks_list <- list(
  callback_early_stopping(
    monitor = "val_accuracy", patience = 2), #                              ➊
  callback_model_checkpoint( #                                              ➋
                            filepath = "checkpoint_path.keras", #           ➌
                            monitor = "val_loss", save_best_only = TRUE) #  ➍
)


model <- get_mnist_model()
model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy") #                                                   ➎

model_history <- model %>% fit(
  train_images, train_labels,
  epochs = 10,
  callbacks = callbacks_list, #                                             ➏
  validation_data = list(val_images, val_labels)) #                         ➐


# ➊ Interrupt training when validation accuracy has stopped improving for two epochs.
# ➋ Save the current weights after every epoch.
# ➌ Path to the destination model file
# ➍ These two arguments mean you won't overwrite the model file unless val_loss has improved, which allows you to keep the best model seen during training.
# ➎ You monitor accuracy, so it should be part of the model's metrics.
# ➏ Callbacks are passed to the model via the callbacks argument in fit(), which takes a list of callbacks. You can pass any number of callbacks.
# ➐ Note that because the callback will monitor validation loss and validation accuracy, you need to pass validation_data to the call to fit().










