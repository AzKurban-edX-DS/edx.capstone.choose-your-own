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
                    cache_file = NULL){
  
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
      
      acc <- mean(y_hat == y0.1.test)
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

predict.dl_model <- function(model,
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
  
  put_log("Function `predict.dl_model`:
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



cnn.binclass.get_prediction_values <- function(preds) {
  # (as.vector(preds) > 0.5) |> as.integer()
  preds %>% `>` (0.5) |> as.integer()
}
