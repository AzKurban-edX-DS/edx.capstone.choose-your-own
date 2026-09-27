# library(keras3)
library(kerastuneR)
library(magrittr)

## Init Directories ------------------------------------------------------------

kt.mnist.dir <- "r/learning/kerastuneR/kt.mnist"

if(!dir.exists(kt.mnist.dir))
  dir.create(kt.mnist.dir)

mnist_prj.dir <- file.path(kt.mnist.dir, "mnist-prj")

if(!dir.exists(mnist_prj.dir))
  dir.create(mnist_prj.dir)

## Tune MNIST Model ------------------------------------------------------------

mnist_data = dataset_fashion_mnist()
c(mnist_train, mnist_test) %<-%  mnist_data
rm(mnist_data)

str(mnist_train)
str(mnist_train$x)
dim(mnist_train$x)
shape(mnist_train$x)

# mnist_train.x = tf$dtypes$cast(mnist_train$x, 'float32') / 255.
mnist_train_x = tf$dtypes$cast(mnist_train$x / 255, 'int32')
str(mnist_train_x)
dim(mnist_train_x)
shape(mnist_train_x)

mnist_train.x = mnist_train_x[seq(6e3),,] # / 255.
str(mnist_train.x)
dim(mnist_train.x)
shape(mnist_train.x)

# mnist_train$x[1,,]

str(mnist_train$y)
dim(mnist_train$y)
shape(mnist_train$y)


# mnist_test$x = tf$dtypes$cast(mnist_test$x, 'float32') / 255.

# x_train = keras3::array_reshape(mnist_train$x, dim = c(6e4,28,28))# [6e3,,]
# x_train = keras3::array_reshape(mnist_train$x[seq(6e3),,], dim = c(6e3,28,28))# 
x_train <- mnist_train.x
x_train <-  tf$dtypes$cast(x.train[seq(6e3),,], 'int32')
# x_train <- np_array(x.train[seq(6e3),,])
str(x_train)
dim(x_train)
shape(x_train)

# # x_test = keras3::array_reshape(mnist_test$x, dim = c(1e4,28,28))
# x_test = keras3::array_reshape(mnist_test$x[seq(1e3),,], dim = c(1e3,28,28))
# # x_test <- np_array(x.test)
# str(x_test)
# dim(x_test)
# shape(x_test)

# y_train <- mnist_train$y
y_train <- mnist_train$y[seq(6e3)]
y_train <- as.array(as.integer(y.train[seq(6e3)]) -1)
stopifnot(max(y_train) == 38)
stopifnot(min(y_train) == 0)

str(y_train)
dim(y_train)
shape(y_train)

# y_test <- mnist_test$y
# y_test <- mnist_test$y[seq(1e3)]
# # y_test <- y.test
# str(y_test)
# dim(y_test)
# shape(y_test)

# mnist_model = function(hp) {
#   
#   model = keras_model_sequential() %>% 
#     layer_flatten(input_shape = c(28,28))
#   for (i in 1:(hp$get('num_layers')) ) {
#     # model %>% layer_dense(32, activation='relu') %>% 
#     #   layer_dense(units = 10, activation='softmax')
#     model %>% layer_dense(128, activation='relu') %>% 
#       layer_dense(units = 39, activation='softmax')
#   } %>% 
#     compile(
#       optimizer = tf$keras$optimizers$Adam(hp$get('learning_rate')),
#       loss = 'sparse_categorical_crossentropy',
#       metrics = 'accuracy') 
#   
#   return(model)
# }


# hp = HyperParameters()
# hp$Choice('learning_rate', c(1e-1, 1e-2, 1e-3, 1e-4))
# hp$Int('num_layers', 2L, 20L)

# tuner = RandomSearch(
#   hypermodel =  dnnb_mcc.sequential,
#   # hypermodel =  mnist_model,
#   max_trials = 5,
#   hyperparameters = hp,
#   tune_new_entries = T,
#   objective = 'val_accuracy',
#   directory = mnist_prj.dir,
#   project_name = 'mnist_space')
# 
# tuner %>% fit_tuner(x = x_train,
#                     y = y_train,
#                     # validation_data = list(x_test,
#                     #                        y_test),
#                     validation_split = 0.2,  # Uses 20% of the data for validation
#                     epochs = 5)
                    
tuner <- dl.tune.hwr_model(dl.build_model = dnnb_mcc.tunable_model,
                           x_train = x_train,
                           y_train = y_train,
                           mnist_prj.dir,
                           project_name = "DL.Basic.Tuner.Probe")

result = kerastuneR::plot_tuner(tuner)
# the list will show the plot and the data.frame of tuning results
result 

# best_5_models = tuner %>% get_best_models(5)
# best_5_models[[1]] %>% plot_keras_model()






