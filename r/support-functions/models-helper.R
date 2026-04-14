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

tune.rf <- function(x, y, mtry, file = NULL){
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)
  
  start <- put_start_date()
  
  if (file.exists(cache_file.path)) {
    put_log("Loading Model Fit Data from cache file: 
%1", cache_file.path)
    
    load(cache_file.path)
    put_log("Train Data list has been loaded from cache.")
    put_end_date(start)
    
  } else {
    
    fit_rf.nzv.mtry11_14 = c(11, 12, 13, 14)
    
    # Time difference of 56.0386 secs
    
    
    fit_rf.nzv.mtry11_14.tuned_result <- lapply( fit_rf.nzv.mtry11_14, function(mtry.val){
      put_log("Tuning `RF` model for `mtry = %1`...", mtry.val)
      start <- put_start_date()
      
      fit <-randomForest(x0.1.train_nzv, 
                         y0.1.train,  
                         mtry = mtry.val, 
                         ntree = 200)
      
      plot(fit)
      
      put_log("The `RF` model has been pre-trained on the dataset: `x0.1.train`
with parameter value: `.mtry = %1`.", mtry.val)
      put_end_date(start)
      
      put_log("Predicting `RF` model on `x0.1.test` for `mtry = %1`...", mtry.val)
      start <- put_start_date()
      
      y_hat <- stats::predict(fit, x0.1.test, type = "response")
      
      put_log("The `RF` Model: Generating predictions task has been completed.")
      
      
      put_log("Validating accuracy of the `RF.mtry9` Model predictions 
made for the `x0.1.test` dataset...")
      
      acc <- mean(y_hat == y0.1.test)
      put_log("The accuracy value is %1", acc)
      put_end_date(start)
      # Time difference of ??? mins
      
      c(mtry=mtry.val, 
        predictions = y_hat,
        err.rate = fit$err.rate,
        accuracy = acc)
    }) 
    
    put_end_date(start)
    #> Time difference of 44.35714 mins
    
    put_log("Saving the model tuning result...")
    
    save(fit_rf.nzv.mtry11_14.tuned_result,
         fit_rf.nzv.mtry11_14,
         file = cache_file.path)
    
    put_log("The Pre-train fit result has been saved to the cache file:
%1.", cache_file.path)
  }
  
  stopCluster(cl)
  stopImplicitCluster()
  
  put_log("Summary of tuned results for the `RF` model:
%1", summary(fit_rf.nzv.mtry11_14.tuned_result),
          capture_output = 1)
  
  put_log("Data structure of tuned results for the `RF` model:
%1", str(fit_rf.nzv.mtry11_14.tuned_result),
          capture_output = 1)
  
  
}