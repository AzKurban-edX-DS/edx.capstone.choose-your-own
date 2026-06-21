learning.keras_tuner.dir <- "r/learning/kerastuneR"
tuner_basic.dir <- file.path(learning.keras_tuner.dir, "tuner-basic")

if(!dir.exists(tuner_basic.dir))
  dir.create(tuner_basic.dir)

# library(kerastuneR)
library(magrittr)

x_data <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
str(x_data)
y_data <-  ifelse(runif(50,0,1) > 0.6, 1L,0L) |> as.matrix()
str(y_data)

x_data2 <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
str(x_data2)
y_data2 <-  ifelse(runif(50,0,1) > 0.6, 1L,0L) |> as.matrix()
str(y_data2)

build_model = function(hp) {
  inputs <- keras_input(c(ncol(x_data)))
  
  outputs <- inputs |>
    layer_dense(units = hp$Int('units',
                               min_value = 32,
                               max_value = 512,
                               step=  32),
                # input_shape = ncol(x_data),
                activation =  'relu') |>
    layer_dense(units = 1, activation = 'softmax')
  
  model <- keras_model(inputs, outputs) 
    
  model |>  compile(
      optimizer = tf$keras$optimizers$Adam(
        hp$Choice('learning_rate',
                  values=c(1e-2, 1e-3, 1e-4))),
      loss = 'binary_crossentropy',
      metrics = 'accuracy')
  return(model)
}

tuner = RandomSearch(
  build_model,
  objective = 'val_accuracy',
  max_trials = 5,
  executions_per_trial = 3,
  directory = tuner_basic.dir,
  project_name = 'helloworld')

tuner |> search_summary()

tuner %>% fit_tuner(x_data,y_data,
                    epochs = 5, 
                    validation_data = list(x_data2,y_data2))

result = kerastuneR::plot_tuner(tuner)
# the list will show the plot and the data.frame of tuning results
result 

# best_5_models = tuner %>% get_best_models(5)
# best_5_models[[1]] %>% plot_keras_model()
