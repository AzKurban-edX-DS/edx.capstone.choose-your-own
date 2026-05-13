#%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Binary Models
#%%%%%%%%%%%%%%%%%%%%%%%%%

#### Define a few parameters to be used in the CNN model --------------------------
# n.output <- 39
batch_size <- 128
num_classes <- 39
epochs <- 100
vld_split <- 0.2

other_labels <- "OL"

#### Open log: Build CNN Model -------------------------------------------------
open_logfile(".build-cnn-model")

#### CNN Model building ------------------------------------------------------------
put_log("Building a set of CNN Binary Classifier Models for the following labels:
%1", capture.output(as.character(y.labels)))
start <- put_start_date()

if(!exists("img28x28mx.array")) {
  stopifnot(file.exists(train.img28x28mx.array.file_path))
  put_log("Loading the Train 28x28 Image Data Array from the backup file...")
  img28x28mx.array <- readRDS(train.img28x28mx.array.file_path)
  put_log("The Train 28x28 Image Data Array has been loaded from the following backup file:
%1", train.img28x28mx.array.file_path)
}
##### Define a CNN model structure ***

hwChar.CNN.binCls.models.backup.path <- file.path(cnn.train.data.path,"cnn.lbl-model.list.rds") 
hwChar.CNN.binCls.models.backup.path

cnn.lbl_model_file.base_name <- "cnn.lbl-model"


cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

cnn.hw_char.models <- lapply(y.labels, function(label) {
  #> Now we define a CNN model with two 2D convolutional layers with max pooling, 
  #> and the 2nd layer with additonal dropout to prevent overfitting. 
  #> Then flatten the output and use two dense layers to connect to the categoires 
  #> of the image. [*]
  
  #label <- y.labels[which(y.labels == "Z")]
  label
  
  put_log("Building model for label `%1` (%2)...", as.character(label), label)
  
  lbl.model_cache.file_path <- file.path(cnn.binary.models, 
                               str_flatten(c(cnn.lbl_model_file.base_name,
                                             label,
                                             as.character(label),
                                             "RData"),
                                           collapse = "."))
  
  lbl.model.file_path <- file.path(cnn.binary.models, 
                                   str_flatten(c(cnn.lbl_model_file.base_name,
                                                 label,
                                                 as.character(label),
                                                 "keras"),
                                               collapse = "."))
  
  lbl.cnn_model <- cnn.create_model.binary_classifier()
  summary(lbl.cnn_model)
  
  put_log("CNN binary classifier model for label `%1` (%2) has been created.", 
          as.character(label), label)
  
  # Similar to DNN model, we need to compile the defined CNN model. [*]
  
  # Compile model
  lbl.cnn_model |> compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_rmsprop(learning_rate = 0.0001),
    metrics = c('accuracy')
  )
  
  put_log("CNN binary classifier model for label `%1` (%2) has been compiled.
Summary of the model:", 
          as.character(label), 
          label,
          capture.output(summary(lbl.cnn_model)))
  
  model_checkpoint.filepath <- 
    file.path(cnn.binary.models,
              str_flatten(c(label,
                            as.character(label),
                            "{epoch:02d}-{val_loss:.2f}.weights.h5"),
                          collapse = "."))
  
  #### Training CNN Model ***
  
  #> Now, we can train the model with our processed data. 
  #> Each epochs's history can be saved to track the progress. 
  #> Please note, as we are not using GPU, it takes a few minutes to finish. 
  #> Please be patient while waiting for the results. 
  #> The training time can be significantly reduced if running on GPU. [*]
  
  set.seed(as.integer(y.labels[y.labels == label]))
  
  lbl.ds.sample.set <- 
    cnn.binclass.sample_sets(img28x28mx.array,
                             label)
  str(lbl.ds.sample.set)
  
  x.cnn_bin.train <- lbl.ds.sample.set$x.train
  str(x.cnn_bin.train)
  
  y.cnn_bin.train <- lbl.ds.sample.set$y.train
  str(y.cnn_bin.train)
  length(y.cnn_bin.train)
  sum(y.cnn_bin.train)
  
  x.cnn_bin.test <- lbl.ds.sample.set$x.test
  y.cnn_bin.test <- lbl.ds.sample.set$y.test
  
  put_log("Training the CNN Model...")
  start <- put_start_date()
  
  #str(x.cnn_bin.train)
  # Train model
  
  cnn.lbl.callbacks <- list(
    # callback_early_stopping(patience = 3, monitor = 'val_loss'),
    callback_model_checkpoint(filepath = model_checkpoint.filepath,
                              monitor = "val_accuracy",
                              mode = max,
                              save_best_only = TRUE,
                              verbose = 1)
    # callback_tensorboard(write_images = TRUE,
    #                      log_dir = cnn.callbacks.tb_logs.path)
  )
  
  
  put_log("Training the (CNN) Binary Classifier Model for label `%1` (%2)...", 
          as.character(label), label)
  
  cnn.1bl.train_history <- lbl.cnn_model |> 
    fit(x.cnn_bin.train, 
        y.cnn_bin.train,
        epochs = epochs,
        batch_size = 50,
        validation_split = vld_split,
        callbacks = cnn.lbl.callbacks
    )
  # acc: 0.8741
  
  put_log("The (CNN) Binary Classifier Model for label `%1` (%2) has been trained.", 
          as.character(label), 
          label)
  
  put_end_date(start)
  
  put_log("Caching the (CNN) Binary Classifier Model for label `%1` (%2)...", 
          as.character(label), 
          label)
  
  save(lbl.model.file_path,
       cnn.1bl.train_history,
       lbl.ds.sample.set,
       file = lbl.model_cache.file_path)
  
  put_log("The (CNN) Binary Classifier Model for label `%1` (%2) has been cached to file:
%3",as.character(label), label,lbl.model_cache.file_path)
  
  put_log("Saving the (CNN) Binary Classifier Model for label `%1`to file...", 
          label)
  
  save_model(lbl.cnn_model,
             filepath = lbl.model.file_path,
             overwrite = TRUE)
  
  put_log("The (CNN) Binary Classifier Model for label `%1` (%2) has been cached to file:
%3",as.character(label), label, lbl.model.file_path)
  
  str(lbl.cnn_model)
  plot(cnn.1bl.train_history)
  
  list(model = lbl.cnn_model,
       saved_model.filepath = lbl.model.file_path,
       lbl.data.cache.path = lbl.model_cache.file_path,
       train_history = cnn.1bl.train_history,
       x.cnn_bin.test = lbl.ds.sample.set$x.test,
       y.cnn_bin.test = lbl.ds.sample.set$y.test,
       label = label)
})
names(cnn.hw_char.models) = as.character(y.labels)

put_log("Saving the set of trained (CNN) Binary Classifier Models info to file...") 

saveRDS(cnn.hw_char.models,
     file = hwChar.CNN.binCls.models.backup.path)

put_log("The set of trained (CNN) Binary Classifier Models have been saved to file:
%1", hwChar.CNN.binCls.models.backup.path)

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

### Close Log ------------------------------------------------------------------
log_close()
#### Open log: Evaluate CNN Model -------------------------------------------------
open_logfile(".evaluate-cnn-model")
#### Evaluating CNN Model ----------------------------------------------

cnn_models.eval.cache_file.path <- file.path(cnn.train.data.path,
                                             "cnn.lbl-models.evaluation.RData")
cnn_models.eval.cache_file.path

put_log("CNN Models Evaluation started for entire set of labels...")
start_eval_time <- put_start_date()


if (file.exists(cnn_models.eval.cache_file.path)) {
  put_log("CNN: Loading the Labeled Binary Classifier Models evalution results from cache...") 
  load(cnn_models.eval.cache_file.path)
  put_log("CNN: the Labeled Binary Classifier Models evalution results 
have been loaded from the cache file:
%1", cnn_models.eval.cache_file.path)
} else {
  cnn.eval_cache.base_name <- "cnn.lbl-model.eval"
  
  if(!exists("cnn.hw_char.models")) {
    if(!file.exists(hwChar.CNN.binCls.models.backup.path))
      stop(str.build("Cache file does not exist: 
%1", hwChar.CNN.binCls.models.backup.path))
    
    cnn.hw_char.models <- readRDS(hwChar.CNN.binCls.models.backup.path)
  }
  
  str(cnn.hw_char.models)
  
  cl <- makeCluster(N_pcCores)
  registerDoParallel(cl)

  evaluation.results <- lapply(y.labels, function(label) {
    put_log("Processing model evaluation for label `%1`...",label)
    start <- put_start_date()

    lbl.model_eval.cache_file.path <- file.path(cnn.eval.cache.path, 
                                 str_flatten(c(cnn.eval_cache.base_name,
                                               label,
                                               as.character(label),
                                               "RData"),
                                             collapse = "."))
    
    put_log("Cache file path: %1", lbl.model_eval.cache_file.path)
    
    if (file.exists(lbl.model_eval.cache_file.path)) {
      put_log("CNN: loading model evaluation data for label `%1` from cache...", label)
      load(lbl.model_eval.cache_file.path)
      put_log("CNN: the model evaluation data for label `%1` have been loaded from the cache file:
%2", label, lbl.model_eval.cache_file.path)
    } else {
      
      lbl.trained_model.list <- cnn.hw_char.models[[label]] 
      #   list(model                = lbl.cnn_model,
      #        saved_model.filepath = lbl.model.file_path,
      #        lbl.data.cache.path  = lbl.model_cache.file_path,
      #        train_history        = cnn.1bl.train_history,
      #        x.cnn_bin.test               = lbl.ds.sample.set$x.test,
      #        y.cnn_bin.test               = lbl.ds.sample.set$y.test,
      #        label                = label)
      
      x.cnn_bin.test <- lbl.trained_model.list$x.cnn_bin.test
      y.cnn_bin.test <- lbl.trained_model.list$y.cnn_bin.test

      #exists("lbl.trained_model.list$train_history")
      plot(lbl.trained_model.list$train_history)

      put_log("Summary of the model for handwritten character `%1`:
%2", label, capture.output(summary(lbl.trained_model.list)))
      
      lbl.cnn_model <- lbl.trained_model.list$model
      str(lbl.cnn_model)

      if(length(lbl.cnn_model) == 0) {
        stopifnot(file.exists(lbl.trained_model.list$saved_model.filepath))
        lbl.cnn_model <- load_model(lbl.trained_model.list$saved_model.filepath)
        stopifnot(length(lbl.cnn_model) > 0)
      }

      str(lbl.cnn_model)
      length(lbl.cnn_model)
      
      put_log("Evaluating the CNN-based Binary  Classifier model for '%1' character...", 
              label)
      lbl.eval.result <- lbl.cnn_model |> evaluate(x.cnn_bin.test, y.cnn_bin.test)
      put_log("Evaluation of the CNN-based Binary  Classifier model for '%1' character has been completed.",
              label)
      
      put_end_date(start)
      
      # model prediction
      put_log("Making predictions using the CNN-based Binary classifier model for the character '%1'...", 
              label)
      start <- put_start_date()
      preds <- lbl.cnn_model |> predict(x.cnn_bin.test) 
      put_log("Prediction generation using the CNN-based Binary classifier model 
for the character '%1' has been completed.",
              label)

      put_log("CNN: Saving model evaluation data for label `%1` to cache...", label)
      save(lbl.trained_model.list,
           lbl.eval.result,
           preds,
           label,
           file = lbl.model_eval.cache_file.path)
      put_log("CNN: the model evaluation data for label `%1` 
have been saved the the cache file:
%2", label, lbl.model_eval.cache_file.path)
    }
    
    plot(lbl.trained_model.list$train_history)

    put_log("The current CNN Model evaluation result:
%1", capture.output(str(lbl.eval.result)))
    # CNN Model evaluation result:
    #   List of 2
    # $ accuracy: num 0.963
    # $ loss    : num 0.752

    put_log("CNN Model for label `%1`: predictions have been constructed.", label)
    put_end_date(start)
    # Time difference of 1.502232 mins
    str(preds)
    head(preds)
    # max(preds)
    
    predictions <- cnn.binclass.get_prediction_values(preds)
    str(predictions)
    sum(predictions)
    
    accuracy <- mean(predictions == y.cnn_bin.test)
    put_log("CNN Model accuracy for label `%1`: %2", label, accuracy)
    # CNN Model accuracy for label `R`: 0.986499530817405
    
    y.cnn_bin.test.factor <- as.factor(y.cnn_bin.test)
    str(y.cnn_bin.test.factor)
    
    predictions.factor <- factor(predictions, levels = levels(y.cnn_bin.test.factor))
    str(predictions.factor)
    
    conf.mx <- confusionMatrix(y.cnn_bin.test.factor, predictions.factor)
    conf.mx
    
    put_log("The confusion matrix for the binary model for recognizing 
the handwritten character '%1' is as follows:

%2", label, capture.output(conf.mx))
    
    
    put_log("The CNN-based Binary  Classifier model evaluation task for '%1' character has been completed.",
            label)
    put_end_date(start)
    
    list(lbl.trained_model.list,
         preds = preds,
         label = label,
         accuracy = accuracy,
         conf.mx)
  })
  names(evaluation.results) <- as.character(y.labels)
  
  put_log("The CNN-based Binary Classifier Model evaluation job has been completed 
for each of the following handwritten characters:
%1", capture.output(as.character(y.labels)))
  
  stopCluster(cl)
  stopImplicitCluster()
  put_end_date(start)
  
lbl_models.accuracies <- sapply(evaluation.results, function(result){
  result$accuracy
})

# names(lbl_models.accuracies) <- as.character(y.labels)
cnn.bin_models.accuracy <- data.frame(label = y.labels, accuracy = lbl_models.accuracies) 
cnn.bin_models.accuracy

  put_log("Saving the CNN-based Binary Classifier Model evaluation results...")
  save(evaluation.results,
       cnn.bin_models.accuracy,
       file = cnn_models.eval.cache_file.path)
  put_log("The CNN-based Binary Classifier model evaluation results have been saved to the following file:
%1", cnn_models.eval.cache_file.path)
}

put_log("The evaluation of the CNN-based Binary Classifier models results in the following accuracies:
%1", capture.output(cnn.bin_models.accuracy))
#    label  accuracy
# 1      # 0.4897910
# 2      $ 0.4902771
# 3      & 0.4913304
# 4      @ 0.4743964
# 5      0 0.4530060
# 6      1 0.4588397
# 7      2 0.4716415
# 8      3 0.4718036
# 9      4 0.4897910
# 10     5 0.4737482
# 11     6 0.4738292
# 12     7 0.4944904
# 13     8 0.4642684
# 14     9 0.4728569
# 15     A 0.4605412
# 16     B 0.4795009
# 17     C 0.4856587
# 18     D 0.4894669
# 19     E 0.4969211
# 20     F 0.5390536
# 21     G 0.4188138
# 22     H 0.4901961
# 23     I 0.4839572
# 24     J 0.5614973
# 25     K 0.5064819
# 26     L 0.4550316
# 27     M 0.4889807
# 28     N 0.4806352
# 29     P 0.4935991
# 30     Q 0.4543834
# 31     R 0.5139362
# 32     S 0.4739102
# 33     T 0.9866310
# 34     U 0.4777994
# 35     V 0.4862259
# 36     W 0.4906822
# 37     X 0.4985416
# 38     Y 0.5098849
# 39     Z 0.4943283

# df.models.acc |>
#   ggplot(aes(x, y)) +
#   geom_line(color = "blue") +
#   geom_point(size = 3)

cnn.bin_models.accuracy |>
  data.plot("Average Accuracy", 
            xname = "label", 
            yname = "accuracy")

put_log("The CNN-based Binary Classifier Model evaluation job has been completed 
for all the handwritten characters with the following results:
%1", capture.output(str(evaluation.results)))

put_end_date(start_eval_time)
### Close Log ------------------------------------------------------------------
log_close()

