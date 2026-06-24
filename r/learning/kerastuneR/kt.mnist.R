
kt.mnist.dir <- "r/learning/kerastuneR/kt.mnist"

if(!dir.exists(kt.mnist.dir))
  dir.create(kt.mnist.dir)

mnist_prj.dir <- file.path(kt.mnist.dir, "mnist-prj")

if(!dir.exists(mnist_prj.dir))
  dir.create(mnist_prj.dir)

# library(keras3)
library(kerastuneR)
library(magrittr)

mnist_data = dataset_fashion_mnist()
c(mnist_train, mnist_test) %<-%  mnist_data
rm(mnist_data)

mnist_train$x = tf$dtypes$cast(mnist_train$x, 'float32') / 255.
str(mnist_train$x)
str(mnist_train$y)


mnist_test$x = tf$dtypes$cast(mnist_test$x, 'float32') / 255.

mnist_train$x = keras3::array_reshape(mnist_train$x, dim = c(6e4,28,28))
mnist_test$x = keras3::array_reshape(mnist_test$x, dim = c(1e4,28,28))


hp = HyperParameters()
hp$Choice('learning_rate', c(1e-1, 1e-3))
hp$Int('num_layers', 2L, 20L)


mnist_model = function(hp) {
  
  model = keras_model_sequential() %>% 
    layer_flatten(input_shape = c(28,28))
  for (i in 1:(hp$get('num_layers')) ) {
    model %>% layer_dense(32, activation='relu') %>% 
      layer_dense(units = 10, activation='softmax')
  } %>% 
    compile(
      optimizer = tf$keras$optimizers$Adam(hp$get('learning_rate')),
      loss = 'sparse_categorical_crossentropy',
      metrics = 'accuracy') 
  
  return(model)
}


tuner = RandomSearch(
  hypermodel =  mnist_model,
  max_trials = 5,
  hyperparameters = hp,
  tune_new_entries = T,
  objective = 'val_accuracy',
  directory = mnist_prj.dir,
  project_name = 'mnist_space')

tuner %>% fit_tuner(x = mnist_train$x,
                    y = mnist_train$y,
                    epochs = 5,
                    validation_data = list(mnist_test$x, mnist_test$y))

result = kerastuneR::plot_tuner(tuner)
# the list will show the plot and the data.frame of tuning results
result 

# best_5_models = tuner %>% get_best_models(5)
# best_5_models[[1]] %>% plot_keras_model()






