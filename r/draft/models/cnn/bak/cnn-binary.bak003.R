#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Binary Classifier Models Set
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#### Define a few parameters to be used in the  model --------------------------
# n.output <- 39
batch_size <- 128
num_classes <- 39
epochs <- 100
vld_split <- 0.2

other_labels <- "OL"

#### Open log: Build CNN Model -------------------------------------------------
open_logfile(".build-cnn-binary.model")

#### Init File Paths -----------------------------------------------------------
put_log("Building a set of CNN Binary Classifier Models for the following classes:
%1", capture.output(as.character(y.labels)))

start <- put_start_date()
stopifnot(file.exists(train.img28x28mx.array.file_path))

put_log("Loading the Train 28x28 Image Data Array Set from the backup file...")
img_mx.set <- readRDS(train.img28x28mx.array.file_path)
put_log("The Train 28x28 Image Data Array Set has been loading from the following file:
%1", train.img28x28mx.array.file_path)
put_log("The Train 28x28 Image Data Array Set structure:
%1", capture.output(str(img_mx.set)))


cnn_bin.models.dat.backup.path <- file.path(data.cnn.binary.dir,
                                            "cnn.binary.associated.data.rds") 
cnn_bin.model.file.base_name <- "cnn.binary-class.model"


cnn_bin.eval.backup.file_path <- file.path(data.cnn.binary.dir,
                                           "cnn.lbl-models.evaluation.RData")

put_log("CNN Models Evaluation started for entire set of labels...")
start_eval_time <- put_start_date()



#### Build, Train & Evaluate Models for each Handwritten Character Class -------

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

cnn_bin.models.dat <- lapply(y.labels, function(label) {

  ##### Preparing a Train & Test Sets for the Class Model  
  {
    set.seed(as.integer(y.labels[y.labels == label]))
    
    cnn_bin.ds_sample.set <- 
      binClass.sample_train_test_sets.x3d(img_mx.set$img28x28mx.array,
                                          img_mx.set$img28x28mx.fpath,
                                          label)
    
    put_log("A sample set with the following structure has been created 
for training & testing the Binary Classifier Model for Class `%1` (%2):
%3", label.char,
            label.idx,
            capture.output(str(cnn_bin.ds_sample.set)))

    x.train <- cnn_bin.ds_sample.set$train_set$x.train
    y.train <- cnn_bin.ds_sample.set$train_set$y.train
    train.files <- cnn_bin.ds_sample.set$train_set$x.files
    str(x.train)
    str(train.files)
    str(y.train)
    length(y.train)
    
    
    # train.groups <- ds.get_classIDs.grouped(x.train)
    # y.train <- train.groups$classID
    
    # y.train.cat <- to_categorical(y.train)
    # dim(y.train.cat)
    
    x.test <- cnn_bin.ds_sample.set$test_set$x.test
    y.test <- cnn_bin.ds_sample.set$test_set$y.test
    test.files <- cnn_bin.ds_sample.set$test_set$x.files
    str(x.test)
    str(test.files)
    str(y.test)
    length(y.test)
    
    # test.groups <- ds.get_classIDs.grouped(x.test)
    # y.test <- test.groups$classID
    
    # y.test.cat <- to_categorical(y.test)
    # dim(y.test.cat)
  
    # rm(cnn_bin.ds_sample.set)
  }

  ##### Defining the Class Model Structure *************************************
  {  
    #> Now we define a CNN model with two 2D convolutional layers with max pooling, 
    #> and the 2nd layer with additonal dropout to prevent overfitting. 
    #> Then flatten the output and use two dense layers to connect to the categoires 
    #> of the image. [*]
    
    
    label.idx <- as.integer(label)
    label.char <- as.character(label)
    
    put_log("Building model for class `%1` (%2)...", label.char, label.idx)
    
    cnn_bin.model.supporting_data.file_path <- file.path(data.cnn.binary.models.dir, 
                                                         str_flatten(c(cnn_bin.model.file.base_name,
                                                                       label.idx,
                                                                       label.char,
                                                                       "RData"),
                                                                     collapse = "."))
    
    bin_model.file_path <- file.path(data.cnn.binary.models.dir, 
                                     str_flatten(c(cnn_bin.model.file.base_name,
                                                   label.idx,
                                                   label.char,
                                                   "keras"),
                                                 collapse = "."))
    
    # bin_model <- cnn.create_model.binary_classifier()
    bin_model <-   keras_model_sequential(shape(28L, 28L, 1L)) |>
      layer_conv_2d(filters = 8L,
                    kernel_size = 2,
                    strides = 1,
                    activation = "relu") |>
      layer_max_pooling_2d(strides = c(2, 2)) |>
      layer_dropout(rate = 0.25) |>
      layer_conv_2d(filters = 16L,
                    kernel_size = 5,
                    strides = 2,
                    activation = "relu") |>
      layer_max_pooling_2d(strides = c(2, 2)) |>
      layer_flatten() |>
      layer_dense(units = 128, activation = "relu") |>
      layer_dropout(rate = 0.3) |>
      layer_dense(units = 1, activation = "sigmoid")

    put_log("CNN binary classifier model has been created for class `%1` (%2).", 
            label.char, label.idx)
    
    # Similar to DNN model, we need to compile the defined CNN model. [*]
    
    # Compile model
    bin_model |> compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_rmsprop(learning_rate = 0.0001),
      metrics = c('accuracy')
    )
    
    put_log("CNN binary classifier model for class `%1` (%2) has been compiled.
Summary of the model:
          
%3", label.char, label.idx, capture.output(summary(bin_model)))
    
    bin_model.checkpoint.file_path <- 
      file.path(data.cnn.binary.models.checkpoints.dir,
                str_flatten(c(label,
                              label.char,
                              "{epoch:02d}-{val_loss:.2f}.keras"),
                            collapse = "."))
  }  
  
  ##### Training CNN-Based Binary Classifier Model ******************************
  {  
    #> Now, we can train the model with our processed data. 
    #> Each epochs's history can be saved to track the progress. 
    #> Please note, as we are not using GPU, it takes a few minutes to finish. 
    #> Please be patient while waiting for the results. 
    #> The training time can be significantly reduced if running on GPU. [*]
    
    cnn.lbl.callbacks <- list(
      # callback_early_stopping(patience = 3, monitor = 'val_loss'),
      callback_model_checkpoint(filepath = bin_model.checkpoint.file_path,
                                monitor = "val_accuracy",
                                mode = max,
                                save_best_only = TRUE,
                                verbose = 1)
      # callback_tensorboard(write_images = TRUE,
      #                      log_dir = cnn.callbacks.tb_logs.path)
    )
    
    
    put_log("Training the (CNN) Binary Classifier Model for Class `%1` (%2)...", 
            label.char, label.idx)
    
    cnn.1bl.train_history <- bin_model |> 
      fit(x.train, 
          y.train,
          epochs = epochs,
          batch_size = 50,
          validation_split = vld_split,
          callbacks = cnn.lbl.callbacks
      )
    # acc: 0.8741
    
    put_log("The (CNN) Binary Classifier Model for Class `%1` (%2) has been trained.", 
            label.char, 
            label.idx)
  }  
  put_end_date(start)
 
  #### Saving Pre-Trained Model Data *******************************************
  {  
    put_log("Backing the (CNN) Binary Classifier Model-related data for Class `%1` (%2)...", 
            label.char, 
            label.idx)
    
    save(bin_model.file_path,
         cnn.1bl.train_history,
         cnn_bin.ds_sample.set,
         file = cnn_bin.model.supporting_data.file_path)
    
    put_log("The (CNN) Binary Classifier Model for Class `%1` (%2) has been backed up to file:
%3",label.char, label,cnn_bin.model.supporting_data.file_path)
    
    put_log("Saving the (CNN) Binary Classifier Model for Class `%1` (%2) to file...", 
            label.char,
            label.idx)
    
    save_model(bin_model,
               filepath = bin_model.file_path,
               overwrite = TRUE)
    
    put_log("The (CNN) Binary Classifier Model for label `%1` (%2) has been cached to file:
%3",label.char, 
            label.idx, 
            bin_model.file_path)
    
  }
  
  str(bin_model)
  plot(cnn.1bl.train_history)
  
  #### Evaluate Pre-trained Model ************************************    
  {
    put_log("Evaluating the CNN-based Binary  Classifier model for '%1' character...", 
            label)
    lbl.eval.result <- bin_model |> evaluate(x.test, y.test)
    put_log("Evaluation of the CNN-based Binary  Classifier model for '%1' character has been completed.",
            label)
    
    # put_end_date(start)
    
    # model prediction
    put_log("Making predictions using the CNN-based Binary classifier model for the character '%1'...", 
            label)
    start <- put_start_date()
    preds <- bin_model |> predict(x.test) 
    put_log("Prediction generation using the CNN-based Binary classifier model 
for the character '%1' has been completed.",
            label)
    
    put_log("CNN: Saving model evaluation data for label `%1` to cache...", label)
    save(lbl.trained_model.list,
         lbl.eval.result,
         preds,
         label,
         file = lbl.model_eval.backup.path)
    put_log("CNN: the model evaluation data for label `%1` 
have been saved the the cache file:
%2", label, lbl.model_eval.backup.path)
  }    
  
  ### Make Predictions & Analyse Results *********************************
  {    
    
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
    
    accuracy <- mean(predictions == y.test)
    put_log("CNN Model accuracy for label `%1`: %2", label, accuracy)
    # CNN Model accuracy for label `R`: 0.986499530817405
    
    y.test.factor <- as.factor(y.test)
    str(y.test.factor)
    
    predictions.factor <- factor(predictions, levels = levels(y.test.factor))
    str(predictions.factor)
    
    conf.mx <- confusionMatrix(y.test.factor, predictions.factor)
    conf.mx
    
    put_log("The confusion matrix for the binary model for recognizing 
the handwritten character '%1' is as follows:

%2", label, capture.output(conf.mx))
    
    
    put_log("The CNN-based Binary  Classifier model evaluation task for '%1' character has been completed.",
            label)
  }
  put_end_date(start)
  
  list(lbl.trained_model.list,
       preds = preds,
       label = label,
       accuracy = accuracy,
       conf.mx)
  
  
  
  list(saved_model.filepath = bin_model.file_path,
       lbl.data.cache.path = cnn_bin.model.supporting_data.file_path,
       train_history = cnn.1bl.train_history,
       x.test = x.test,
       y.test = y.test,
       label = label)
})

names(cnn_bin.models.dat) = as.character(y.labels)

put_log("Saving the set of trained (CNN) Binary Classifier Models info to file...") 

saveRDS(cnn_bin.models.dat,
     file = cnn_bin.models.dat.backup.path)

put_log("The set of trained (CNN) Binary Classifier Models have been saved to file:
%1", cnn_bin.models.dat.backup.path)

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

### Close Log ------------------------------------------------------------------
log_close()
#### Open log: Evaluate CNN Model ----------------------------------------------
open_logfile(".evaluate-cnn-model")
#### CNN Binary Models Evaluation ----------------------------------------------

if (file.exists(cnn_bin.eval.backup.file_path)) {
  put_log("CNN: Loading the Labeled Binary Classifier Models evalution results from cache...") 
  load(cnn_bin.eval.backup.file_path)
  put_log("CNN: the Labeled Binary Classifier Models evalution results 
have been loaded from the cache file:
%1", cnn_bin.eval.backup.file_path)
} else {
  cnn.eval_cache.base_name <- "cnn.lbl-model.eval"
  
  if(!exists("cnn_bin.models.dat")) {
    if(!file.exists(cnn_bin.models.dat.backup.path))
      stop(str.build("Cache file does not exist: 
%1", cnn_bin.models.dat.backup.path))
    
    cnn_bin.models.dat <- readRDS(cnn_bin.models.dat.backup.path)
  }
  
  str(cnn_bin.models.dat)
  
  # cl <- makeCluster(N_pcCores)
  # registerDoParallel(cl)

  evaluation.results <- lapply(y.labels, function(label) {
    
#### Evaluate Pre-trained Model ************************************    
    
    put_log("Processing model evaluation for label `%1`...",label)
    start <- put_start_date()

    lbl.model_eval.backup.path <- file.path(data.cnn.binary.models.evaluation.dir, 
                                 str_flatten(c(cnn.eval_cache.base_name,
                                               label,
                                               as.character(label),
                                               "RData"),
                                             collapse = "."))
    
    put_log("Cache file path: %1", lbl.model_eval.backup.path)
    
    if (file.exists(lbl.model_eval.backup.path)) {
      put_log("CNN: loading model evaluation data for label `%1` from cache...", label)
      load(lbl.model_eval.backup.path)
      put_log("CNN: the model evaluation data for label `%1` have been loaded from the cache file:
%2", label, lbl.model_eval.backup.path)
    } else {
      # Load pre-trained model data
      
      lbl.trained_model.list <- cnn_bin.models.dat[[label]] 
      #   list(model                = bin_model,
      #        saved_model.filepath = bin_model.file_path,
      #        lbl.data.cache.path  = cnn_bin.model.supporting_data.file_path,
      #        train_history        = cnn.1bl.train_history,
      #        x.test               = cnn_bin.ds_sample.set$x.test,
      #        y.test               = cnn_bin.ds_sample.set$y.test,
      #        label                = label)
      
      x.test <- lbl.trained_model.list$x.test
      y.test <- lbl.trained_model.list$y.test

      #exists("lbl.trained_model.list$train_history")
      plot(lbl.trained_model.list$train_history)

      put_log("Summary of the model for handwritten character `%1`:
%2", label, capture.output(summary(lbl.trained_model.list)))
      
      bin_model <- lbl.trained_model.list$model
      str(bin_model)

      if(length(bin_model) == 0) {
        stopifnot(file.exists(lbl.trained_model.list$saved_model.filepath))
        bin_model <- load_model(lbl.trained_model.list$saved_model.filepath)
        stopifnot(length(bin_model) > 0)
      }

      str(bin_model)
      length(bin_model)
      
    }
    
    plot(lbl.trained_model.list$train_history)

  })
  names(evaluation.results) <- as.character(y.labels)
  
  put_log("The CNN-based Binary Classifier Model evaluation job has been completed 
for each of the following handwritten characters:
%1", capture.output(as.character(y.labels)))
  
  # stopCluster(cl)
  # stopImplicitCluster()
  # put_end_date(start)
  
lbl_models.accuracies <- sapply(evaluation.results, function(result){
  result$accuracy
})

# names(lbl_models.accuracies) <- as.character(y.labels)
cnn.bin_models.accuracy <- data.frame(label = y.labels, accuracy = lbl_models.accuracies) 
cnn.bin_models.accuracy

  put_log("Saving the CNN-based Binary Classifier Model evaluation results...")
  save(evaluation.results,
       cnn.bin_models.accuracy,
       file = cnn_bin.eval.backup.file_path)
  put_log("The CNN-based Binary Classifier model evaluation results have been saved to the following file:
%1", cnn_bin.eval.backup.file_path)
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

