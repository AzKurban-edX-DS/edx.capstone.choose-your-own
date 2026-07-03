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

dl_basic.model.sequential = function(hp) {
  
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

build.dl_basic.model <- function(hp) {
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

dl_basic.tunable_model <- function(hp,
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

predict.dl_basic.model <- function(model,
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
  #> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>
  
  predictions <- preds.ts |> op_argmax(2)
  
  put_log("Function `predict.dl_basic.model`:
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

predicted_probs2classes <- function(x, classes.factor) {
  sapply(seq(nrow(x)), function(i) {
    classes.factor[which.max(x[i,])]
  })
}

cnn.binclass.get_prediction_values <- function(preds) {
  # (as.vector(preds) > 0.5) |> as.integer()
  preds %>% `>` (0.5) |> as.integer()
}

## Analysis & Visualization ----------------------------------------------------

plot.ROC.curves <- function(targets,
                            predicted_probabilities) {

  put_log("Function `plot.ROC.curves`:
Calculating a ROC curve for each class of the Multiclass Classifier (MCC)...")
  roc_curves <- calc.roc_curves(targets,
                                bdl.preds,
                                Y.Labels)
  
  put_log("Function `plot.ROC.curves`:
The per-class ROC curve calculation of the MCC has been completed.")
  
  put_log("Function `plot.ROC.curves`:
Plotting the ROC curves...")
  plot(roc_curves[[1]], 
       main = "ROC Curves for the `Basic Deep Learning Multiclass Classifier` Model")
  
  for (class.idx in 2:N.classes) {
    lines(roc_curves[[class.idx]], col = class.idx)
  }
  
  return(roc_curves)
}

calc.roc_curves <- function(targets,
                            predicted_probabilities,
                            class_labels) {
  categorical_targets <- 
    length(shape(targets)) == 2 && shape(y.test.cat)[2] == N.classes
  
  lapply(class_labels, function(class) {
    class.idx <- as.integer(class)
    
    if(categorical_targets) {
      bin_labels <- targets[, class.idx]
    } else {
      stopifnot(class(targets) == "factor" && 
                  sum(as.character(Y.Labels) == levels(targets)) == N.classes)
      
      bin_labels <- as.integer(targets == class)
    }
    roc_curve <- roc(bin_labels, predicted_probabilities[, as.integer(class)])
  })
}

create_confusion_matrix <- function(targets, 
                                    predictions,
                                    create_plot = FALSE,
                                    palette = "Greens",
                                    font_counts = font(size = 3,
                                                       color = "red"),
                                    add_normalized = FALSE,
                                    add_col_percentages = FALSE,
                                    add_row_percentages = FALSE) {
  put_log("Function `create_confusion_matrix`:
Creating confusion matrix...")
  start <- put_start_date()
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)

  conf.mx <- confusion_matrix(as.character(targets),
                                           as.character(predictions))
  put_log("Function `create_confusion_matrix`:
The confution matrix has been created:
%1", capture.output(conf.mx))

  if(create_plot){
    put_log("Function `create_confusion_matrix`:
Plotting confusion matrix, please wait...")
    plot.mx <- plot_confusion_matrix(conf.mx,
                                     palette = palette,
                                     font_counts = font_counts,
                                     add_normalized = add_normalized,
                                     add_col_percentages = add_col_percentages,
                                     add_row_percentages = add_row_percentages)
  }

  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  
  if(create_plot) {
    list(conf.mx = conf.mx,
         mx.plot = plot.mx)
  } else {
    conf.mx
  }
}

plot_bars.accuracy.by_class <- function(classes.factor,
                                        class.accuracies,
                                        title.prefix = NULL,
                                        .title = "Classifier Model: Class-wise Evaluation Result",
                                        .color = "black",
                                        .fill = "steelblue") {
  if(!is.null(title.prefix))
    .title = paste(title.prefix, .title)
  
  data.frame(class = classes.factor,
             accuracy = class.accuracies) |>
    ggplot(mapping = aes(x = class,
                         y = accuracy)) +
    geom_col(fill = .fill,
             color = .color) +
    labs(x = "Handwritten Character Class",
         y = "Accuracy",
         title = .title) +
    scale_y_continuous(labels = scales::label_percent(accuracy = 1),
                       expand = c(0, 0, 0.005, 0))
}


## Utility Functions -----------------------------------------------------------

# Multiclass Classifier: Class-wise Accuracy
MCClassifier.accuracy.by_class <- function(classes.factor,
                                         y.test,
                                         predicted.values) {
  y.test.idx <- seq_len(length(y.test))

  sapply(classes.factor, function(class) {
    idx <- y.test.idx[y.test == class]
    n <- length(idx)

    accuracy <- mean(predicted.values[idx] == class)
    
    put_log("Function `multiclass.accuracy.by_class`:
Accuracy for the class `%1` (of size %2) is %3.",
            class, n, accuracy) 
    accuracy
  }) |> matrix(ncol = 1, dimnames = list(class = Y.Labels, "accuracy")) 
}

