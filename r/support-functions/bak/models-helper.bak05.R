#%%%%%%%%%%%%%%%%%%%%%%%%
# Models Helper Functions 
#%%%%%%%%%%%%%%%%%%%%%%%%

## Building Machine Learning Models --------------------------------------------

train.kNN_PCA <- function(x, 
                          y, 
                          k.values, 
                          cv.number = 10,
                          cv.p = 0.75,
                          pca.thresh = 0.95,
                          cacheFile
                          ){

  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  train_knn_pca <- caret::train(x, y, method = "knn", 
                                preProcess = c("nzv", "pca"),
                                trControl = trainControl("cv", number = cv.number, p = cv.p,
                                                         preProcOptions = list(thresh = pca.thresh)),
                                tuneGrid = data.frame(k = k.values))
  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  train_knn_pca
}

tune.rf <- function(x, 
                    y, 
                    x.test,
                    y.test,
                    mtry = NA, 
                    n.tree = 200,
                    cache_root = NULL,
                    cache_file = NULL) {
  
  local_root <- NULL
  local_cache.path <- NULL
  
  if(!is.null(cache_root)){
    if(!dir.exists(cache_root))
      dir.create(cache_root)
    
    local_root <- file.path(cache_root, "tune.rf.cache")
    cache_file <- file.path(cache_root, cache_file)
    
    if(!dir.exists(local_root))
      dir.create(local_root)
  }

  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  start <- put_start_date()
  
  if (!is.null(cache_file) && file.exists(cache_file)) {
    put_log("Function `tune.rf`:
Loading Model Fit Data from cache file: 
%1", cache_file)
    
    tuned.info <- readRDS(cache_file)
    mtry.tuned_result <- tuned.info$tuned_result
    mtry <- tuned.info$mtry
    rm(tuned.info)
    
    put_log("Function `tune.rf`:
Train Data list has been loaded from cache.")
    put_end_date(start)
    
  } else {

    # mtry.tuned_result <- lapply(mtry, function(mtry.val) {
    #   print_log("`mtry.val` is NA: %1",is.na(mtry.val))
    #   mtry.val
    # })
    
    mtry.tuned_result <- lapply(mtry, function(mtry.val){
      start <- put_start_date()
      
      if (!is.null(local_root)) {
        m <- ifelse(is.na(mtry.val), "default", mtry.val)
        local_cache <- str.build("rf.fit.ntree%1.mtry%2.rds", n.tree, m)
        local_cache.path <- file.path(local_root, local_cache)
      }
      
      if (!is.null(local_cache.path) && file.exists(local_cache.path)) {
        put_log("Function `tune.rf`:
Loading RF Tuning Fit Data from local cache file: 
%1", local_cache.path)
        
        fit <- readRDS(local_cache.path)
        put_log("Function `tune.rf`:
RF Tuning Fit Data has been loaded from cache.")
        put_end_date(start)
        
      } else {
        

        if (is.na(mtry.val)) {
          put_log("Function `tune.rf`:
  Tuning `RF` model for the default value of `mtry`...")
  
          fit <-randomForest(x, 
                             y,  
                             ntree = n.tree)
        } else {
          put_log("Function `tune.rf`:
  Tuning `RF` model for `mtry = %1`...", mtry.val)
          
          fit <-randomForest(x, 
                             y,  
                             mtry = mtry.val, 
                             ntree = n.tree)
        }
        
        if (!is.null(local_cache.path)){
          
          put_log("Function `tune.rf`:
Caching the model tuning fit result...")

          saveRDS(fit, file = local_cache.path)

          put_log("Function `tune.rf`:
The model tuning fit result has been cached to file:
%1.", local_cache.path)
        }
      }
      
      plot(fit)
      
      put_log("Function `tune.rf`:
The `RF` model has been trained with parameter value: `.mtry = %1`.", 
              mtry.val)
      put_end_date(start)
      
      put_log("Function `tune.rf`:
Summary of training result for mtry = %1:
%2", mtry.val, str(fit), cupture_output = 2)
      
      
      put_log("Function `tune.rf`:
Predicting `RF` model on `x.test` for `mtry = %1`...", mtry.val)
      start <- put_start_date()
      
      y_hat <- stats::predict(fit, x.test, type = "response")
      
      put_log("Function `tune.rf`:
The `RF` Model: Generating predictions task has been completed.")
      
      put_log("Function `tune.rf`:
Summary of prediction results for mtry = %1:
%2", mtry.val, str(y_hat), cupture_output = 2)
      
      
      put_log("Function `tune.rf`:
Validating accuracy of the `RF.mtry9` Model predictions 
made for the `x.test` dataset...")
      
      acc <- mean(y_hat == y.test)
      put_log("Function `tune.rf`:
The accuracy value is %1", acc)
      put_end_date(start)
      # Time difference of ??? mins
      
      list(in.mtry=mtry.val,
           fit = fit,
           predictions = y_hat,
           err.rate = fit$err.rate,
           accuracy = acc)
    }) 
    put_end_date(start)
    
    accuracy <- sapply(mtry.tuned_result, function(result) result$accuracy)
    max.acc.idx <- which.max(accuracy)
    
    best_result <- mtry.tuned_result[[max.acc.idx]]
    
    put_log("Function `tune.rf`:
Data structure of the best result of the model tuning:
%1", str(best_result),
            capture_output = 1)
    
    if (!is.null(cache_file)){
      
      put_log("Function `tune.rf`:
Saving the model tuning result...")
      saveRDS(list(tuned_result = mtry.tuned_result,
                   mtry = mtry),
              file = cache_file)
      put_log("Function `tune.rf`:
The Pre-train fit result has been saved to the cache file:
%1.", cache_file)
    }
  }
  
  stopCluster(cl)
  stopImplicitCluster()
  
  put_log("Function `tune.rf`:
Data structure of tuned results for the `RF` model:
%1", str(mtry.tuned_result),
          capture_output = 1)
  
  mtry.tuned_result
}

x.binarize <- function(x) {
  nzv <- nearZeroVar(x)
  x.nzv <- x[, -nzv]
  (x > 0.5)*1
}

dnn_basic.model.sequential = function(hp) {
  
  model = keras_model_sequential() %>% 
    layer_flatten(input_shape = c(28,28))
  
  for (i in 1:(hp$get('num_layers')) ) {
    model %>% layer_dense(128, activation='relu') %>% 
      layer_dense(units = 39, activation='softmax')
  } %>% 
    compile(
      optimizer = tf$keras$optimizers$Adam(hp$get('learning_rate')),
      loss = 'sparse_categorical_crossentropy',
      metrics = 'accuracy') 
  
  return(model)
}

build.dnn_basic.model <- function(hp) {
  n.inputs <- 28*28
 
  model_inputs <- layer_input(shape = c(n.inputs))
  
  model_outputs <- model_inputs |>
    layer_dense(units = hp$Int("units",
                               min_value = 64,
                               max_value = n.inputs,
                               step=  32), 
                activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    # layer_dense(units = n.hl.units, activation = "relu") |>
    # layer_dropout(rate = 0.25) |> 
    layer_dense(units = N.classes, activation = "softmax")
  
  
  model <- keras_model(model_inputs, model_outputs)

  model |> compile(
    loss = "categorical_crossentropy",
    optimizer = tf$keras$optimizers$Adam(
      hp$Choice('learning_rate',
                values=c(1e-2, 1e-3, 1e-4))),
    metrics = "accuracy"
  )
  
  summary(model)
  model
}

dl.tune.hwr_model <- function(dl.build_model,
                              x_train,
                              y_train,
                              tuner_project.dir,
                              tuner_checkpoints.file,
                              input_shape = c(28, 28),
                              dropout.rate = 0.2,
                              learning_rate = c(1e-1, 1e-2, 1e-3, 1e-4),
                              min_layers = 2L,
                              max_layers = 20L,
                              min_units = 39,
                              max_units = 784,
                              units.tune_step = 32,
                              max_trials = 5,
                              tune_new_entries = TRUE,
                              objective = 'val_accuracy',
                              project_name = 'DL.ModelTuner',
                              validation_split = 0.2,
                              validation_data = NULL,
                              epochs = 100) {
  dl.tune_model(dl.build_model,
                x_train,
                y_train,
                tuner_project.dir,
                tuner_checkpoints.file,
                input_shape,
                N.classes,
                dropout.rate,
                learning_rate,
                min_layers,
                max_layers,
                min_units,
                max_units,
                units.tune_step,
                max_trials,
                tune_new_entries,
                objective,
                project_name,
                validation_split,
                validation_data,
                epochs)
}

dl.tune_model <- function(dl.build_model,
                          x_train,
                          y_train,
                          tuner_project.dir,
                          tuner_checkpoints.file,
                          input_shape = c(28, 28),
                          n.outputs,
                          dropout.rate = 0.2,
                          learning_rate = c(1e-1, 1e-2, 1e-3, 1e-4),
                          min_layers = 2L,
                          max_layers = 20L,
                          min_units = 39,
                          max_units = 784,
                          units.tune_step = 32,
                          max_trials = 5,
                          tune_new_entries = TRUE,
                          objective = 'val_accuracy',
                          project_name = 'DL.ModelTuner',
                          validation_split = 0.2,
                          validation_data = NULL,
                          epochs = 100) {
  
  stopifnot(dir.exists(tuner_project.dir))
  hp = HyperParameters()
  
  # Choice of one value among a predefined set of possible values.
  # Choice(name, values, ordered = NULL, default = NULL, parent_name = NULL, parent_values = NULL)
  hp$Choice('learning_rate', learning_rate)
  
  hp$Int('num_layers', 
         min_layers,
         max_layers)
  
  callback_list <- list(
    callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
    callback_model_checkpoint(filepath = tuner_checkpoints.file,
                              monitor = "val_loss",
                              save_best_only = TRUE,
                              verbose = 1)
  )
  
  
  for (i in seq(max_layers)) {
    hp$Int(paste0("units_", i),
           min_value = min_units,
           max_value = max_units,
           step = units.tune_step)    
  }
  
  tuner = RandomSearch(
    hypermodel =  function(hp) dl.build_model(hp,
                                              input_shape,
                                              n.outputs,
                                              dropout.rate),
    max_trials = max_trials,
    hyperparameters = hp,
    tune_new_entries = tune_new_entries,
    objective = objective,
    directory = tuner_project.dir,
    project_name = project_name)
  
  if(is.null(validation_data)){
    tuner %>% fit_tuner(x = x_train,
                        y = y_train,
                        callbacks = callback_list,
                        validation_split = validation_split,
                        epochs = epochs)
  } else {
    tuner %>% fit_tuner(x = x_train,
                        y = y_train,
                        callbacks = callback_list,
                        validation_data = validation_data,
                        epochs = epochs)
  }
  
  tuner
}

cnn_mcc.tunable_model <- function(hp,
                                   input_shape = c(28L, 28L, 1),
                                   n.outputs,
                                   dropout.rate = 0.2) {
#> References:
#> Hyperparameter tuning with Keras Tuner
#> https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html  
  
  model_inputs <- layer_input(shape = input_shape)
  #layer <- model_inputs |> layer_flatten()
  
  # Tune the NUMBER OF LAYERS dynamically
  for (i in 1:hp$get('conv_blocs')) {
    
    layer <- layer |>
      layer_conv_2d(filters = hp$get(paste0("filters_", i)),
                    kernel_size = c(3L, 3L), 
                    # strides = list(1L, 1L),
                    activation = "relu") |>
      # layer_max_pooling_2d() |>
      layer_max_pooling_2d(pool_size = c(2, 2)) #|>
      # Good practice: add dropout to prevent deep layers from overfitting
#      layer_dropout(rate = dropout.rate) 
  }
  
  model_outputs <- layer |> 
    layer_dense(units = n.outputs, activation = "softmax")
  
  model <- keras_model(model_inputs, model_outputs)
  
  # For `loss` argument, Use sparse_categorical_crossentropy if labels are integers
  model|>compile(loss = "sparse_categorical_crossentropy",
                 optimizer =  keras3::optimizer_adamax(hp$get('learning_rate')),
                 metrics = "accuracy"
  )
  
  return(model)
}

dnn_mcc.tunable_model <- function(hp,
                                  input_shape = c(28, 28),
                                  n.outputs,
                                  dropout.rate = 0.2) {
  
  model_inputs <- layer_input(shape = input_shape)
  layer <- model_inputs |> layer_flatten()

  # Tune the NUMBER OF LAYERS dynamically
  # The tuner will try anywhere from 1 to 4 hidden layers
  for (i in 1:hp$get('num_layers')) {
    
    layer <- layer |>
      layer_dense(units = hp$get(paste0("units_", i)),
                  activation = "relu") |>
      # Good practice: add dropout to prevent deep layers from overfitting
      layer_dropout(rate = dropout.rate) 
  }
  
  model_outputs <- layer |> 
    layer_dense(units = n.outputs, activation = "softmax")

  model <- keras_model(model_inputs, model_outputs)
  
  # For `loss` argument, Use sparse_categorical_crossentropy if labels are integers
  model|>compile(loss = "sparse_categorical_crossentropy",
    optimizer =  keras3::optimizer_adamax(hp$get('learning_rate')),
    metrics = "accuracy"
  )
  
  return(model)
}

#' @param class.labels Multiclass Classifier Class Labels   
predict.dnn_basic.model <- function(model,
                             x.test,
                             class.labels) {
  put_log("Evaluating DL Model...")
  start <- put_start_date()
  preds <- model |> predict(x.test) 
  put_log("DL Model evaluation has been completed.")
  put_end_date(start)
  # Time difference of 1.502232 mins
  # dim(preds)
  
  colnames(preds) <- class.labels
  # head(preds[,1:5])
  
  preds.ts <- as_tensor(preds)
  str(preds.ts)
  #' <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>
  
  predictions <- preds.ts |> op_argmax(2)
  
  put_log("Function `predict.dnn_basic.model`:
Predictions have been constructed:
%1", capture.output(str(predictions)))
  put_end_date(start)
  
  predictions$numpy()
}

cnn.create_model.binary_classifier <- function(img.width = 28L,
                                               img.height = 28L,
                                               filters1 = 8L,
                                               filters2 = 16L,
                                               kernal1.size = c(2.2),
                                               kernal2.size = 5,
                                               conv_2d.strides1 = 1,
                                               conv_2d.strides2 = 2,
                                               max_pooling_2d.strides1 = c(2, 2),
                                               max_pooling_2d.strides2 = c(2,2),
                                               pooling_2d.dropout.rate = 0.25,
                                               dense.dropout.rate = 0.3,
                                               dense.units = 128,
                                               output.classes = 1,
                                               output.activation = "sigmoid") {
  cnn.create_model(img.width,
                   img.height,
                   filters1,
                   filters2,
                   kernal1.size,
                   kernal2.size,
                   conv_2d.strides1,
                   conv_2d.strides2,
                   max_pooling_2d.strides1,
                   max_pooling_2d.strides2,
                   pooling_2d.dropout.rate,
                   dense.dropout.rate,
                   dense.units,
                   output.classes,
                   output.activation)
}

cnn.create_model <- function(img.width,
                             img.height,
                             filters1,
                             filters2,
                             kernal1.size,
                             kernal2.size,
                             conv_2d.strides1,
                             conv_2d.strides2,
                             max_pooling_2d.strides1,
                             max_pooling_2d.strides2,
                             pooling_2d.dropout.rate,
                             dense.dropout.rate,
                             dense.units,
                             output.classes,
                             output.activation) {
  
  keras_model_sequential(shape(img.width, img.height, 1L)) |>
    layer_conv_2d(filters = filters1,
                  kernel_size = kernal1.size, 
                  strides = conv_2d.strides1,
                  activation = "relu") |>
    layer_max_pooling_2d(strides = max_pooling_2d.strides1) |>
    layer_dropout(rate = pooling_2d.dropout.rate) |>
    layer_conv_2d(filters = filters2, 
                  kernel_size = kernal2.size,
                  strides = conv_2d.strides2,
                  activation = "relu") |>
    layer_max_pooling_2d(strides = max_pooling_2d.strides2) |>
    layer_flatten() |>
    layer_dense(units = dense.units, activation = "relu") |>
    layer_dropout(rate = dense.dropout.rate) |>
    layer_dense(units = output.classes, activation = output.activation)
}

#' @param class.labels Multiclass Classifier Class Labels   
predicted_probs2classes <- function(x, class.labels) {
  sapply(seq(nrow(x)), function(i) {
    class.labels[which.max(x[i,])]
  })
}

cnn.binclass.get_prediction_values <- function(preds) {
  # (as.vector(preds) > 0.5) |> as.integer()
  preds %>% `>` (0.5) |> as.integer()
}

## Analysis & Visualization ----------------------------------------------------
### Init Plot Args -------------------------------------------------------------

#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param predicted.probabilities Predicted probability values 
#' obtained as a result of the model's evaluation.
#' @param predicted.values Predicted values obtained as a result 
#' of the model's evaluation.
#' @param model_type Type of model: in `This Project`, it can be one of the following:
#' - `Multiclass Classifier`;
#' - `Binary Classifier`.
#' @param alg_name Name of the algorithm used for the model training.
#' @param roc.plot_title `ROC Curves` Plot title.
#' @param roc.plot_name `ROC Curves` Plot name
#' @param pca.plot_title Multiclass Classifier `Per-Class Accuracy` (`PCA`) plot title.
#' @param pca.plot_name `PCA` plot name.
#' @param pca.color `PCA` plot line color.
#' @param pca.fill `PCA` plot fill color.
#' @param cm.plot_title `Confusion Matrix` (`CM`) plot title.
#' @param cm.plot_name `CM` plot name.
#' @returns An object of the `modelEvalResultPlotArgs` class
create.plot_args <- function(targets,
                             predicted.probabilities,
                             predicted.values,
                             plots_dat.file,
                             model_type = "Multiclass Classifier",
                             alg_name = NULL,
                             roc.plot_title = NULL,
                             roc.plot_name = "ROC Curves",
                             pca.plot_title = NULL,
                             pca.plot_name = "Class-wise Evaluation Result",
                             pca.color = "black",
                             pca.fill = "steelblue",
                             cm.plot_title = NULL,
                             cm.plot_name = "Confusion Matrix",
                             cm.palette = "Greens",
                             cm.font.size = 3,
                             cm.font.color = "red",
                             cm.add_normalized = FALSE,
                             cm.add_col_percentages = FALSE,
                             cm.add_row_percentages = FALSE,
                             cm.print.plot_object = FALSE,
                             cm.print.image = FALSE,
                             cm.export.img_file = NULL,
                             cm.backup.file = NULL) {
  
  stopifnot(class(predicted.values) == "factor", 
            sum(levels(predicted.values) == levels(Y.Labels)) == N.classes)
  
  cm.export2image.validate(cm.print.image,
                           cm.export.img_file)
  
  return (structure(list(targets = targets,
                         predicted.probabilities = predicted.probabilities,
                         predicted.values = predicted.values,
                         model_type = model_type,
                         alg_name = alg_name,
                         roc.plot_title = roc.plot_title,
                         roc.plot_name = roc.plot_name,
                         pca.plot_title = pca.plot_title,
                         pca.plot_name = pca.plot_name,
                         pca.fill = pca.fill,
                         pca.color = pca.color,
                         cm.plot_title = cm.plot_title,
                         cm.plot_name = cm.plot_name,
                         cm.palette = cm.palette,
                         cm.font.size = cm.font.size,
                         cm.font.color = cm.font.color,
                         cm.add_normalized = cm.add_normalized,
                         cm.add_col_percentages = cm.add_col_percentages,
                         cm.add_row_percentages = cm.add_row_percentages,
                         cm.print.plot_object = cm.print.plot_object,
                         cm.print.image = cm.print.image,
                         cm.export.img_file = cm.export.img_file,
                         cm.backup.file = cm.backup.file), 
                    class = "modelEvalResultPlotArgs"))
}

#### `init.plots_args` Generic Function ----------------------------------------
#' (Using `S3 Method Dispatch` Mechanism)

#' Generic method for plotting `Per-Class Accuracy (PCA) Bar Charts` based on the 
#' model evaluation results passed as the function arguments.
init.plots_args <- function(x, ...) {
  UseMethod("init.plots_args")
}

#' Initialize an object, containing values to pass as arguments to visualization functions to for plots by the following script:
#' [Shared  Model Evaluation Results Visualization](r/models/model-visualization.shared.R)
#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param predicted.probabilities Predicted probability values 
#' obtained as a result of the model's evaluation.
#' @param predicted.values Predicted values obtained as a result 
#' of the model's evaluation.
#' @param model_type Type of model: in `This Project`, it can be one of the following:
#' - `Multiclass Classifier`;
#' - `Binary Classifier`.
#' @param alg_name Name of the algorithm used for the model training.
#' @param roc.plot_title `ROC Curves` Plot title.
#' @param roc.plot_name `ROC Curves` Plot name
#' @param pca.plot_title Multiclass Classifier `Per-Class Accuracy` (`PCA`) plot title.
#' @param pca.plot_name `PCA` plot name.
#' @param pca.color `PCA` plot line color.
#' @param pca.fill `PCA` plot fill color.
#' @param cm.plot_title `Confusion Matrix` (`CM`) plot title.
#' @param cm.plot_name `CM` plot name.
init.plots_args.default <- function(targets,
                                    predicted.probabilities,
                                    predicted.values,
                                    plots_dat.file,
                                    model_type = "Multiclass Classifier",
                                    alg_name = NULL,
                                    roc.plot_title = NULL,
                                    roc.plot_name = "ROC Curves",
                                    pca.plot_title = NULL,
                                    pca.plot_name = "Per-Class Accuracy",
                                    pca.color = "black",
                                    pca.fill = "steelblue",
                                    cm.plot_title = NULL,
                                    cm.plot_name = "Confusion Matrix",
                                    cm.palette = "Greens",
                                    cm.font.size = 3,
                                    cm.font.color = "red",
                                    cm.add_normalized = FALSE,
                                    cm.add_col_percentages = FALSE,
                                    cm.add_row_percentages = FALSE,
                                    cm.print.plot_object = FALSE,
                                    cm.print.image = FALSE,
                                    cm.export.img_file = NULL,
                                    cm.backup.file = NULL) {
  
  
  create.plot_args(targets = targets,
                   predicted.probabilities = predicted.probabilities,
                   predicted.values = predicted.values,
                   model_type = model_type,
                   alg_name = alg_name,
                   roc.plot_title = roc.plot_title,
                   roc.plot_name = roc.plot_name,
                   pca.plot_title = pca.plot_title,
                   pca.plot_name = pca.plot_name,
                   pca.color = pca.color,
                   pca.fill = pca.fill,
                   cm.plot_title = cm.plot_title,
                   cm.plot_name = cm.plot_name,
                   cm.palette = cm.palette,
                   cm.font.size = cm.font.size,
                   cm.font.color = cm.font.color,
                   cm.add_normalized = cm.add_normalized,
                   cm.add_col_percentages = cm.add_col_percentages,
                   cm.add_row_percentages = cm.add_row_percentages,
                   cm.print.plot_object = cm.print.plot_object,
                   cm.print.image = cm.print.image,
                   cm.export.img_file = cm.export.img_file,
                   cm.backup.file = cm.backup.file)
}


init.plots_args.character <- function(plots_dat.file) {
  stopifnot(file.exists(plots_dat.file))

  put_log("Function `init.plots_args.character`:
Loading the model-related plots input data object from the backup file...")
  plots_args <- readRDS(plots_dat.file)
  
  put_log("Function `init.plots_args.character`:
The model-related plots input data object has been loaded from the following file:
%1", plots_dat.file)
  
  return(plots_args)
}

### Plotting ROC Curves --------------------------------------------------------
# References:
# https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=Precision%2Drecall%20curves%20are%20created,x%2Daxis%20across%20all%20thresholds.
# https://www.geeksforgeeks.org/machine-learning/roc-curves-for-multiclass-classification-in-r/

#' Calculates the `ROC Curves` data based on the model evaluation result values 
#' passed in the function arguments.
#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param predicted_probabilities Predicted probability values 
#' obtained as a result of the model's evaluation.
#' @param class.labels Multiclass Classifier Class Labels   
#' @returns An object of the `rocCurvesDat` class 
#' representing the `ROC Curves` data.
calc.roc_curves <- function(targets,
                            predicted_probabilities,
                            class.labels) {
  targets
  categorical_targets <- 
    length(shape(targets)) == 2 && 
    shape(targets)[2] == length(levels(class.labels))
  
  put_log("Function `calc.roc_curves`:
Calculating the ROC curves data...")
  lapply(class.labels, function(class) {
    class.idx <- as.integer(class)
    
    if(categorical_targets) {
      bin_labels <- targets[, class.idx]
    } else {
      stopifnot(class(targets) == "factor" && 
                  sum(levels(targets) != levels(class.labels)) == 0)
      
      bin_labels <- as.integer(targets == class)
    }
    roc_curve <- roc(bin_labels, predicted_probabilities[, as.integer(class)])
  })
}

#### `plot.ROC_curves` Generic Function ----------------------------------------
#' (Using `S3 Method Dispatch` Mechanism)


#' Generic method for plotting `ROC Curves` based on the model evaluation results 
#' passed as the function arguments.
plot.ROC_curves <- function(x, ...) {
  UseMethod("plot.ROC_curves")
}
# plot.ROC_curves <- function(x) {
# }


#' @details
#' Plots `ROC Curves` based on the model evaluation results passed 
#' as the `modelEvalResultPlotArgs` class object in the `x` argument.
#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.ROC_curves.default()` function.
#' @param x An object of the `modelEvalResultPlotArgs` class 
#' representing the model evaluation results.
#' @returns An object of the `rocCurvesDat` class representing the `ROC Curves` data.
plot.ROC_curves.modelEvalResultPlotArgs <- function(x) {
  
  put_log("Function `plot.ROC_curves.modelEvalResultPlotArgs`:
Using S3 Method Dispatch
to call the `plot.ROC_curves.default()` function...")
  # Plots `ROC Curves` and returns an object of the `rocCurvesDat` class
  plot.ROC_curves(x$targets,
                  x$predicted.probabilities,
                  x$roc.plot_title,
                  x$roc.plot_name,
                  x$model_type,
                  x$alg_name)
}

#' @details
#' Plots `ROC Curves` based on the visual representation data of the model 
#' evaluation results passed as the `plotsDat` class object in the `x` argument.
#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.ROC_curves.rocCurvesDat()` function.
#' @param x An object of the `plotsDat` class.
#' @returns An object of the `rocCurvesDat` class representing the `ROC Curves` data.
plot.ROC_curves.plotsDat <- function(x) {
  
  put_log("Function `plot.ROC_curves.plotsDat`:
Using `S3 Method Dispatch` to call the `plot.ROC_curves.rocCurvesDat()` function...")
  plot.ROC_curves(x$ROC)
  
  # An object of the `rocCurvesDat` class. 
  return(x$ROC)
}

#' @details
#' Plots `ROC Curves` based on the model evaluation result values 
#' passed as the function arguments.
#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.ROC_curves.rocCurvesDat()` function.
#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param predicted_probabilities Predicted probability values 
#' obtained as a result of the model's evaluation.
#' @param title Plot title.
#' @param plot_name Plot name.
#' @param model_type Type of model: in `This Project`, it can be one of the following:
#' - `Multiclass Classifier`;
#' - `Binary Classifier`.
#' @param alg_name Name of the algorithm used for the model training.
#' @returns An object of the `rocCurvesDat` class representing the `ROC Curves` data.
plot.ROC_curves.default <- function(targets,
                                    predicted_probabilities,
                                    title = NULL,
                                    plot_name = "ROC Curves",
                                    model_type = "Multiclass Classifier",
                                    alg_name = NULL) {
  plot_title <- build.plot_title(title,
                                 plot_name,
                                 model_type,
                                 alg_name)
  
  put_log("Function `plot.ROC_curves.default`:
Calculating a ROC curve for each class of the Multiclass Classifier (MCC)...")
  roc_curves <- calc.roc_curves(targets,
                                predicted_probabilities,
                                Y.Labels)
  
  roc_curves.dat <- structure(list(roc_curves = roc_curves,
                                   title = plot_title), class = "rocCurvesDat")
  
  put_log("Function `plot.ROC_curves.default`:
Using S3 Method Dispatch
to call the `plot.ROC_curves.rocCurvesDat()` function...")
  plot.ROC_curves(roc_curves.dat)
 
  # An object of the `rocCurvesDat` class. 
  return(roc_curves.dat)
}

#' @details
#' Plots `ROC Curves` based on the visual representation data of the model 
#' evaluation results passed as the `rocCurvesDat` class object in the `x` argument.
#' @param x An object of the `rocCurvesDat` class.
plot.ROC_curves.rocCurvesDat <- function(x) {
  put_log("Function `plot.ROC_curves.rocCurvesDat`:
Plotting the ROC curves...")
  plot(x$roc_curves[[1]], 
       main = x$title)
  
  for (label.idx in 2:N.classes) {
    lines(x$roc_curves[[label.idx]], col = label.idx)
  }
}

### Plotting Per-Class Accuracy (PCA) Bar Charts -------------------------------

#' Calculates `PCA` values as part of the `Multiclass Classifier` (`MCC`) model's 
#' evaluation process.
#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param predicted.values Predicted values obtained as a result 
#' of the model's evaluation.
#' @param class.labels `MCC` Class Labels   
#' @returns `Single-column matrix` containing a list of the `PCA` values 
#' for the `MCC` model being evaluated.
MCClassifier.accuracy.by_class <- function(targets,
                                           predicted.values,
                                           class.labels) {
  stopifnot(class(class.labels) == "factor")
  
  stopifnot(class(predicted.values) == "factor" && 
              sum(levels(predicted.values) != levels(class.labels)) == 0)

  categorical_targets <- 
    length(shape(targets)) == 2 && 
    shape(targets)[2] == length(levels(class.labels))
  
  sapply(class.labels, function(class) {
    if(categorical_targets) {
      targets.idx <- seq_len(nrow(y.test.cat))      
      idx <- targets.idx[targets[, as.integer(class)] == 1]
    } else {
      stopifnot(class(targets) == "factor" && 
                  sum(levels(targets) != levels(class.labels)) == 0)
      
      targets.idx <- seq_len(length(targets))
      idx <- targets.idx[targets == class]
    }

    n <- length(idx)
    accuracy <- mean(predicted.values[idx] == class)
    
    put_log("Function `MCClassifier.accuracy.by_class`:
Accuracy for the class `%1` (of size %2) is %3.",
            class, n, accuracy) 
    accuracy
  }) |> 
    matrix(ncol = 1, dimnames = list(class = class.labels, "accuracy")) 
}

#' Plots `PCA Bar Chart` based on the model's 
#' `Per-Class Accuracy` (`PCA`) values.
#' @param class.accuracies Model's `PCA` values
#' @param class.labels Multiclass Classifier Class Labels   
#' @param .title Plot title.
#' @param color Plot line color.
#' @param fill Plot fill color.
plot_bars.accuracy.by_class <- 
  function(class.accuracies,
           class.labels,
           .title = "Classifier Model: Class-wise Evaluation Result",
           .color = "black",
           .fill = "steelblue") {
    
    data.frame(class = class.labels,
               accuracy = class.accuracies) |>
      ggplot(mapping = aes(x = accuracy,
                           y = class)) +
      geom_col(fill = .fill,
               color = .color) +
      labs(x = "Accuracy",
           y = "Handwritten Character Class Labels",
           title = .title) +
      scale_x_continuous(labels = scales::label_percent(accuracy = 1),
                         expand = c(0, 0, 0.005, 0))
  }

#### `barPlot.accuracy_by_class` Generic Function ------------------------------
#' (Using `S3 Method Dispatch` Mechanism)

#' Generic method for plotting `Per-Class Accuracy (PCA) Bar Charts` based on the 
#' model evaluation results passed as the function arguments.
barPlot.accuracy_by_class <- function(x, ...) {
  UseMethod("barPlot.accuracy_by_class")
}


#' @details
#' Plots `PCA Bar Chart` based on the visual representation data of the model 
#' evaluation results (`Per-Class Accuracy` OF `PCA`) passed as a `PCA` property 
#' of the `plotsDat` class object in the `x` argument.
#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `barPlot.accuracy_by_class.perClassAccValues()` function.
#' @param x An object of the `plotsDat` class.
#' @returns An object of the `perClassAccValues` class representing 
#' the `PCA Bar Chart` data.
barPlot.accuracy_by_class.plotsDat <- function(x) {
    barPlot.accuracy_by_class(x$PCA)
}


#' @details
#' Plots `PCA Bar Charts` based on the model evaluation results 
#' passed as the `modelEvalResultPlotArgs` class object in the `x` argument.
#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `barPlot.accuracy_by_class.default()` function.
#' @param x An object of the `modelEvalResultPlotArgs` class 
#' representing the model evaluation results.
#' @returns An object of the `perClassAccValues` class representing 
#' the `PCA Bar Chart` data.
barPlot.accuracy_by_class.modelEvalResultPlotArgs <- function(x) {
  
  stopifnot(class(x$predicted.values) == "factor" && 
              sum(levels(x$predicted.values) != levels(Y.Labels)) == 0)
  
  barPlot.accuracy_by_class(x$targets,
                            x$predicted.values,
                            x$pca.plot_title,
                            x$pca.plot_name,
                            x$model_type,
                            x$alg_name,
                            x$pca.color,
                            x$pca.fill)
}

#' @details
#' Plots `PCA Bar Chart` based on the model evaluation result values 
#' passed as the function arguments.
#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `barPlot.accuracy_by_class.perClassAccValues()` function.
#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param predicted.values Predicted values obtained as a result 
#' of the model's evaluation.
#' @param title Plot title.
#' @param plot_name Plot name.
#' @param model_type Type of model: in `This Project`, it can be one of the following:
#' - `Multiclass Classifier`;
#' - `Binary Classifier`.
#' @param alg_name Name of the algorithm used for the model training.
#' @returns An object of the `perClassAccValues` class representing 
#' the `PCA Bar Chart` data.
barPlot.accuracy_by_class.default <- function(targets,
                                              predicted.values,
                                              title = NULL,
                                              plot_name = "Class-wise Evaluation Result",
                                              model_type = "Multiclass Classifier",
                                              alg_name = NULL,
                                              color = "black",
                                              fill = "steelblue") {
  
  stopifnot(class(predicted.values) == "factor" && 
              sum(levels(predicted.values) != levels(Y.Labels)) == 0)
  
  plot_title <- build.plot_title(title ,
                                 plot_name,
                                 model_type,
                                 alg_name)
  
  per_class.accuracy <- MCClassifier.accuracy.by_class(targets,
                                                       predicted.values,
                                                       Y.Labels)
  x <- structure(list(acc.by_class = per_class.accuracy,
                      title = plot_title,
                      color = color,
                      fill = fill),
                 class = "perClassAccValues")
  
  barPlot.accuracy_by_class(x)
}

#' @details
#' Plots `PCA Bar Chart` based on the visual representation data of the model 
#' evaluation results passed as the `perClassAccValues` class object 
#' in the `x` argument.
#' @param x An object of the `perClassAccValues` class.
#' @returns An object of the `perClassAccValues` class.
barPlot.accuracy_by_class.perClassAccValues <- function(x) {
    
  put_log("Function `barPlot.accuracy_by_class.perClassAccValues`: 
Plotting a bar chart of the model's per-class accuracies...")
  bar_plot <- plot_bars.accuracy.by_class(x$acc.by_class,
                                          Y.Labels,
                                          .title = x$title,
                                          .color = x$color,
                                          .fill = x$fill)
  print(bar_plot)
  return(x)
}

### Plotting Confusion Matrix --------------------------------------------------

#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param pred_values Predicted values obtained as a result of the model's evaluation.
#' @returns An object of the `ConfMxPlotArgs` class
plot.cm.create_args <- function(targets,
                                pred_values,
                                title,
                                palette = "Greens",
                                font.size = 3,
                                font.color = "red",
                                add_normalized = FALSE,
                                add_col_percentages = FALSE,
                                add_row_percentages = FALSE,
                                cm.print.plot_object = FALSE,
                                cm.print.image = FALSE,
                                cm.export.img_file = NULL,
                                cm.backup.file = NULL) {
  
  stopifnot(class(pred_values) == "factor", 
            sum(levels(pred_values) == levels(Y.Labels)) == N.classes)
  
  cm.export2image.validate(cm.print.image,
                           cm.export.img_file)

  args <- list(targets = targets,
               pred_values = pred_values,
               title = title,
               palette = palette,
               font.size = font.size,
               font.color = font.color,
               add_normalized = add_normalized,
               add_col_percentages = add_col_percentages,
               add_row_percentages = add_row_percentages,
               cm.print.plot_object = cm.print.plot_object,
               cm.print.image = cm.print.image,
               cm.export.img_file = cm.export.img_file,
               cm.backup.file = cm.backup.file)
  
  return(structure(args, class = "ConfMxPlotArgs"))
}

#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param pred_values Predicted values obtained as a result of the model's evaluation.
#' @returns An object of the `ConfMxDat` class
build_confusion_matrix.plot_data <- function(targets,
                                             pred_values,
                                             title,
                                             palette = "Greens",
                                             font.size = 3,
                                             font.color = "red",
                                             add_normalized = FALSE,
                                             add_col_percentages = FALSE,
                                             add_row_percentages = FALSE,
                                             cm.print.plot_object = FALSE,
                                             cm.print.image = FALSE,
                                             cm.export.img_file = NULL,
                                             cm.backup.file = NULL) {

  stopifnot(class(pred_values) == "factor", 
            sum(levels(pred_values) == levels(Y.Labels)) == N.classes)
  
  cm.export2image.validate(cm.print.image,
                           cm.export.img_file)
  
  put_log("Function `build_confusion_matrix.plot_data`:
Creating a confusion matrix object from the model evaluation results 
suitable for visualization using the `cvms` package...")
  cm <- create.confusion_matrix(targets,
                                pred_values)

  return(structure(list(cm = cm,
                        title = title,
                        palette = palette,
                        font.size = font.size,
                        font.color = font.color,
                        add_normalized = add_normalized,
                        add_col_percentages = add_col_percentages,
                        add_row_percentages = add_row_percentages,
                        cm.print.plot_object = cm.print.plot_object,
                        cm.print.image = cm.print.image,
                        cm.export.img_file = cm.export.img_file,
                        cm.backup.file = cm.backup.file), 
                   class = "ConfMxDat"))
}

#' Creates a confusion matrix object from the model evaluation results 
#' using the `cvms` package.
#' @param targets Target values of the `Test Set` used for the model's evaluation.
create.confusion_matrix <- function(targets,
                                    predicted.values) {
  
  stopifnot(class(predicted.values) == "factor", 
            sum(levels(predicted.values) == levels(Y.Labels)) == N.classes)
  
  categorical_targets <- 
    length(shape(targets)) == 2 && 
    shape(targets)[2] == N.classes
  
  
  if(categorical_targets) {
    y.idx <- sapply(seq_len(nrow(targets)), 
                    function(i) which(y.test.cat[i,] == 1))
    
    y <- Y.Labels[y.idx]
    
  } else {
    stopifnot(class(targets) == "factor" && 
                sum(levels(targets) == levels(Y.Labels)) == N.classes)
    y <- targets
  }
  
  put_log("Function `create.confusion_matrix`:
Creating a confusion matrix object from the model evaluation results 
using the `cvms` package...")
  cvms::confusion_matrix(as.character(y),
                   as.character(predicted.values))
}

#' Builds a plot object for the Confusion Matrix.
#' @param x An object of the `ConfMxDat` class. 
#' @returns An object of the `ConfMxPlot` class.
build.ConfMxPlot.object <- function(x) {
  
  
  start <- put_start_date()
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  put_log("Function `plot.confusion_matrix.ConfMxDat`: 
Creating a visual representation of the confusion matrix using the `cvms` package...")
  cm.chart <- plot_confusion_matrix(x$cm,
                                    palette = x$palette,
                                    font_counts = font(size = x$font.size,
                                                       color = x$font.color),
                                    add_normalized = x$add_normalized,
                                    add_col_percentages = x$add_col_percentages,
                                    add_row_percentages = x$add_row_percentages) +
    labs(title = x$title)
  
  put_end_date(start)
  return(structure(list(cm.chart = cm.chart,
                        cm.dat = x$cm,
                        cm.print.plot_object = x$cm.print.plot_object,
                        cm.print.image = x$cm.print.image,
                        cm.export.img_file = x$cm.export.img_file,
                        cm.backup.file = x$cm.backup.file), 
                   class = "ConfMxPlot"))
}

#' Validates the consistency of the Confusion Matrix Visualization-related 
#' functions' arguments.
#' @param cm.print.image (Boolean) Determines whether to print the image - 
#' the result of the export of the confusion matrix. 
#' @param cm.export.img_file A connection or the path name of the file 
#' where the image is to be saved.
cm.export2image.validate <- function(cm.print.image,
                                     cm.export.img_file){
  if(cm.print.image && is.null(cm.export.img_file))
    stop("Confusion Matrix image cannot be printed since the image file path
is not provided: 
`cm.print.image` is `TRUE` but `cm.export.img_file` is `NULL`!")
  
}

#### `plot.confusion_matrix` Generic Function ------------------------------
#' (Using `S3 Method Dispatch` Mechanism)

#' Generic method
plot.confusion_matrix <- function(x, ...) {
  UseMethod("plot.confusion_matrix")
}

#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.confusion_matrix.default()` function.
plot.confusion_matrix.modelEvalResultPlotArgs <- function(x) {
  
  stopifnot(class(x$predicted.values) == "factor", 
            sum(levels(x$predicted.values) == levels(Y.Labels)) == N.classes)
  
  put_log("Function `plot.confusion_matrix.modelEvalResultPlotArgs`:
Using `S3 Method Dispatch` to call the `plot.confusion_matrix.default()` function...")
  plot.confusion_matrix(x$targets,
                        x$predicted.values,
                        x$cm.plot_title,
                        x$cm.plot_name,
                        x$model_type,
                        x$alg_name,
                        x$cm.palette,
                        x$cm.font.size,
                        x$cm.font.color,
                        x$cm.add_normalized,
                        x$cm.add_col_percentages,
                        x$cm.add_row_percentages,
                        x$cm.print.plot_object,
                        x$cm.print.image,
                        x$cm.export.img_file,
                        x$cm.backup.file)
}

#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.confusion_matrix.ConfMxPlotArgs()` function.
#' @param targets Target values of the `Test Set` used for the model's evaluation.
#' @param pred_values Predicted values obtained as a result of the model's evaluation.
#' @param title Plot title.
#' @param plot_name Plot name.
#' @param model_type Type of model: in `This Project`, it can be one of the following:
#' - `Multiclass Classifier`;
#' - `Binary Classifier`.
#' @param alg_name Name of the algorithm used for the model training.
plot.confusion_matrix.default <- function(targets,
                                          pred_values,
                                          title = NULL,
                                          plot_name = "Confusion Matrix",
                                          model_type = "Multiclass Classifier",
                                          alg_name = NULL,
                                          palette = "Greens",
                                          font.size = 3,
                                          font.color = "red",
                                          add_normalized = FALSE,
                                          add_col_percentages = FALSE,
                                          add_row_percentages = FALSE,
                                          cm.print.plot_object = FALSE,
                                          cm.print.image = FALSE,
                                          cm.export.img_file = NULL,
                                          cm.backup.file = NULL) {

  stopifnot(class(pred_values) == "factor", 
            sum(levels(pred_values) == levels(Y.Labels)) == N.classes)
  
  cm.export2image.validate(cm.print.image,
                           cm.export.img_file)
  
  plot_title <- build.plot_title(title,
                                 plot_name,
                                 model_type,
                                 alg_name)
  
  put_log("Function `plot.confusion_matrix.default`:
Creating an object of the `ConfMxPlotArgs` class, 
representing the argument list for the helper function that plots the confusion matrix...")
  args <- plot.cm.create_args(targets = targets,
                              title = plot_title,
                              pred_values = pred_values,
                              palette = palette,
                              font.size = font.size,
                              font.color = font.color,
                              add_normalized = add_normalized,
                              add_col_percentages = add_col_percentages,
                              add_row_percentages = add_row_percentages,
                              cm.print.plot_object = cm.print.plot_object,
                              cm.print.image = cm.print.image,
                              cm.export.img_file = cm.export.img_file,
                              cm.backup.file = cm.backup.file)
  
  put_log("Function `plot.confusion_matrix.default`:
Using S3 Method Dispatch
to call the `plot.confusion_matrix.ConfMxPlotArgs()` function...")
  plot.confusion_matrix(args)
}

#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.confusion_matrix.ConfMxDat()` function.
#' @param x An object of the `ConfMxPlotArgs` class.
plot.confusion_matrix.ConfMxPlotArgs <- function(x) {
  put_log("Function `plot.confusion_matrix.ConfMxPlotArgs`: 
Creating an object of the `ConfMxDat` class, that contains a confusion matrix 
from the model evaluation results, and other values used for visualization, 
using the `cvms` package...")
  cm.dat <- build_confusion_matrix.plot_data(x$targets,
                                             x$pred_values,
                                             title = x$title,
                                             palette = x$palette,
                                             font.size = x$font.size,
                                             font.color = x$font.color,
                                             add_normalized = x$add_normalized,
                                             add_col_percentages = x$add_col_percentages,
                                             add_row_percentages = x$add_row_percentages,
                                             cm.print.plot_object = x$cm.print.plot_object,
                                             cm.print.image = x$cm.print.image,
                                             cm.export.img_file = x$cm.export.img_file,
                                             cm.backup.file = x$cm_data.backup.file)
  put_log("Using S3 Method Dispatch
to call the `plot.confusion_matrix.ConfMxDat()` function...")
  plot.confusion_matrix(cm.dat)
  
}

#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.confusion_matrix.ConfMxPlot()` function.
#' @param x An object of the `ConfMxDat` class.
plot.confusion_matrix.ConfMxDat <- function(x) {

  put_log("Function `plot.confusion_matrix.ConfMxDat`:
Building a `ConfMxPlot` class object - a visual representation 
of the confusion matrix from the model evaluation results...")
  cm.plot <- build.ConfMxPlot.object(x)
  
  put_log("Using S3 Method Dispatch
to call the `plot.confusion_matrix.ConfMxPlot()` function...")
  plot.confusion_matrix(cm.plot)
}

#' Optionally plots the Confusion Matrix and saves it to the backup file. 
#' @param x An object of the `ConfMxPlot` class.
plot.confusion_matrix.ConfMxPlot <- function(x) {
  
  if(!is.null(x$cm.export.img_file)) {
    put_log("Function `plot.confusion_matrix.ConfMxPlot`: 
Exporting the Confusion Matrix Plot object to an image file...")
    
    ggsave(filename = x$cm.export.img_file, 
           plot = x$cm.chart)
    
    put_log("Function `plot.confusion_matrix.ConfMxPlot`: 
The Confusion Matrix Plot image has been saved to the following file:
  %1", x$cm.export.img_file)
  }
  
  if (!is.null(x$cm.backup.file)) {
    put_log("Function `plot.confusion_matrix.ConfMxPlot`: 
Saving the confusion matrix plot object in the backup file...")
    
    saveRDS(x$cm.chart, 
            file = x$cm.backup.file)
    
    put_log("Function `plot.confusion_matrix.ConfMxPlot`: 
The confusion matrix plot object has been backed up in the following file:
`%1`", x$cm.backup.file)
  }
  
  if(x$cm.print.plot_object) {
    # # Clear any stuck graphics devices
    while(!is.null(dev.list())) dev.off()
    # graphics.off() 
    # gc()
    # 
    # # Open a clean external window (use windows() on Windows, x11() on Linux/Mac)
    # dev.new()
    # Sys.sleep(6)
    
    assign("conf_mx.chart.tmp", x$cm.chart, envir = .GlobalEnv)
    
    grid::grid.newpage()
    grid::grid.draw(ggplotGrob(conf_mx.chart.tmp))
    
    stopCluster(cl)
    stopImplicitCluster()
    
    rm(conf_mx.chart.tmp, pos = .GlobalEnv)
  } else if(x$cm.print.image) {
    stopifnot(file.exists(x$cm.export.img_file))
    plot_image(x$cm.export.img_file)
  }
  
  return(x)
}

#' Note: The function uses the `S3 Method Dispatch` internally 
#' to call the `plot.confusion_matrix.()` function.
#' @param x An object of the `plotsDat` class.
plot.confusion_matrix.plotsDat <- function(x) {

  put_log("Function `plot.confusion_matrix.plotsDat`:
Using S3 Method Dispatch
to call the `plot.confusion_matrix.ConfMxPlot()` function...")
  plot.confusion_matrix(x$CM)
}

### Utility Functions -----------------------------------------------------------

#' Builds a plot title. 
#' @details
#' Note: If the value of the `title` argument is not `NULL`, the function returns 
#' the value of that argument, ignoring the other arguments' values.
#' @param title Plot title.
#' @param plot_name Plot name.
#' @param model_type Type of model: in `This Project`, it can be one of the following:
#' - `Multiclass Classifier`;
#' - `Binary Classifier`.
#' @param alg_name Name of the algorithm used for the model training.
build.plot_title <- function(title = NULL,
                             plot_name = "Visualization",
                             model_type = "Multiclass Classifier",
                             alg_name = NULL) {
  if(is.null(title)){
    title <- paste(plot_name,
                        "for the",
                        model_type,
                        "Model")

    if(!is.null(alg_name)){
      title <- paste(title,
                          str.build("Based on the %1 Algorithm", 
                                    alg_name))
    }
  }
  
  title
}


