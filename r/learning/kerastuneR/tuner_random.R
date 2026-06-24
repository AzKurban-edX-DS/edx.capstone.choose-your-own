library(keras3)
# Make sure you have the 'kerastuner' package/backend installed

build_model <- function(hp) {
  model <- keras_model_sequential()
  
  # First layer
  model|>layer_dense(
    units = hp$Int("units_input", min_value = 32, max_value = 256, step = 32),
    activation = "relu",
    input_shape = c(num_features)
  )
  
  # Tune the NUMBER OF LAYERS dynamically
  # The tuner will try anywhere from 1 to 4 hidden layers
  for (i in 1:hp$Int("num_layers", min_value = 1, max_value = 4)) {
    model|>layer_dense(
      units = hp$Int(paste0("units_", i), min_value = 32, max_value = 128, step = 32),
      activation = "relu"
    )
    # Good practice: add dropout to prevent deep layers from overfitting
    model|>layer_dropout(rate = 0.2) 
  }
  
  # Output layer for multiclass classification
  model|>layer_dense(units = num_classes, activation = "softmax")
  
  model|>compile(
    optimizer = "adam",
    loss = "categorical_crossentropy", # Use sparse_categorical_crossentropy if labels are integers
    metrics = c("accuracy")
  )
  
  return(model)
}

# Initialize a Random Search Tuner
tuner <- tuner_random(
  hypermodel = build_model,
  objective = "val_accuracy",
  max_trials = 10,
  directory = "my_tuning_dir"
)

# Run the search to find the optimal layer configuration
tuner|>search_matrix(X_train, y_train, epochs = 20, validation_split = 0.2)
best_hps <- tuner|>get_best_hyperparameters(num_trials = 1)
