dl.tune_model <- function(dnn_mcc.tunable_model,
                          x_train,
                          y_train,
                          dnn_mcc.tuner.dir,
                          dnn_mcc.tuner.checkpoint.file_path,
                          
                          
                          
                          dnn_mcc.tuner.input_shape = c(28, 28),
                          N.classes,
                          dnn_mcc.dropout.rate = 0.2,
                          dnn_mcc.learning_rate = c(1e-1, 1e-2, 1e-3, 1e-4),
                          dnn_mcc.min_layers = 2L,
                          dnn_mcc.max_layers = 20L,
                          dnn_mcc.min_units = 39,
                          dnn_mcc.max_units = 784,
                          dnn_mcc.units.tune_step = 32,
                          dnn_mcc.max_trials = 5,
                          dnn_mcc.tune_new_entries = TRUE,
                          dnn_mcc.objective = 'val_accuracy',
                          dnn_mcc.validation_split = 0.2,
                          dnn_mcc.validation_data = NULL,
                          dnn_mcc.epochs = 100
                          
                          
                          
                          ) {
  
  dnn_mcc.tuner.input_shape <- c(28, 28)
  dnn_mcc.dropout.rate <- 0.2
  dnn_mcc.learning_rate <- c(1e-1, 1e-2, 1e-3, 1e-4)
  dnn_mcc.min_layers <- 2L
  dnn_mcc.max_layers <- 20L
  dnn_mcc.min_units <- 39
  dnn_mcc.max_units <- 784
  dnn_mcc.units.tune_step <- 32
  dnn_mcc.max_trials <- 5
  dnn_mcc.tune_new_entries <- TRUE
  dnn_mcc.objective <- 'val_accuracy'
  dnn_mcc.validation_split <- 0.2
  dnn_mcc.validation_data <- NULL
  dnn_mcc.epochs <- 100
  
  
  hp = HyperParameters()
  
  # Choice of one value among a predefined set of possible values.
  # Choice(name, values, ordered = NULL, default = NULL, parent_name = NULL, parent_values = NULL)
  hp$Choice('dnn_mcc.learning_rate', dnn_mcc.learning_rate)
  
  hp$Int('num_layers', 
         dnn_mcc.min_layers,
         dnn_mcc.max_layers)
  
  callback_list <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = dnn_mcc.tuner.checkpoint.file_path,
                              monitor = "val_loss",
                              save_best_only = TRUE,
                              verbose = 1)
  )
  
  
  for (i in seq(dnn_mcc.max_layers)) {
    hp$Int(paste0("units_", i),
           min_value = dnn_mcc.min_units,
           max_value = dnn_mcc.max_units,
           step = dnn_mcc.units.tune_step)    
  }
  
  dnn_mcc.tuner = RandomSearch(
    hypermodel =  function(hp) dnn_mcc.tunable_model(hp,
                                              dnn_mcc.tuner.input_shape,
                                              N.classes,
                                              dnn_mcc.dropout.rate),
    dnn_mcc.max_trials = dnn_mcc.max_trials,
    hyperparameters = hp,
    dnn_mcc.tune_new_entries = dnn_mcc.tune_new_entries,
    dnn_mcc.objective = dnn_mcc.objective,
    directory = dnn_mcc.tuner.dir,
    project_name = project_name)
  
  if(is.null(dnn_mcc.validation_data)){
    dnn_mcc.tuner |> fit_tuner(x = x_train,
                        y = y_train,
                        callbacks = callback_list,
                        dnn_mcc.validation_split = dnn_mcc.validation_split,
                        dnn_mcc.epochs = dnn_mcc.epochs)
  } else {
    dnn_mcc.tuner |> fit_tuner(x = x_train,
                        y = y_train,
                        callbacks = callback_list,
                        dnn_mcc.validation_data = dnn_mcc.validation_data,
                        dnn_mcc.epochs = dnn_mcc.epochs)
  }
  
  dnn_mcc.tuner
}

