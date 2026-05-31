#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Binary Classifier (CNN BCC) Models Set
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Define CNN BCC-specific parameters ------------------------------------------
cnn.bin.batch_size <- 128
cnn.bin.epochs <- 100
cnn.bin.vld_split <- 0.2

## Init File Paths ------------------------------------------------------------
open_logfile(".build-cnn-binary.model")

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

cnn_bin.eval_backup.base_name <- "cnn-binary.model.eval"


put_log("CNN Models Evaluation started for entire set of Classes...")
start_eval_time <- put_start_date()



## Build, Train & Evaluate Models for each Handwritten Character Class -------

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

cnn_bin.models.dat <- lapply(y.labels, function(label) {

  label.idx <- as.integer(label)
  label.char <- as.character(label)
  
  put_log("Building model for class `%1` (%2)...", label.char, label.idx)
  
  model.supporting_data.file_path <- file.path(data.cnn.binary.models.dir, 
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
  
  bin_model_eval.backup.path <- file.path(data.cnn.binary.models.evaluation.dir, 
                                          str_flatten(c(cnn_bin.eval_backup.base_name,
                                                        label,
                                                        as.character(label),
                                                        "RData"),
                                                      collapse = "."))
  if(file.exists(bin_model.file_path)) {
    put_log("Loading pre-trained CNN-Based Binary Classifier Model for Class `%1`...",
            label.char)
    bin_model <- load_model(bin_model.file_path)
    put_log("The CNN-Based Binary Classifier Model has been loaded from the backup file:
%1", bin_model.file_path)
    
    if(file.exists(model.supporting_data.file_path)){
      put_log("Loading the CNN-Based Binary Classifier Model Train History for Class `%1`...",
              label.char)
      load(model.supporting_data.file_path)
      put_log("The CNN-Based Binary Classifier Model History for Class `%1` has been loaded from the following file:
%2", label.char, model.supporting_data.file_path)
    } else {
      warning("The CNN-Based Binary Classifier Model History backup does not exist for class `%1`:
%2", label.char, model.supporting_data.file_path)
    }
  } else {
    ### Preparing a Train & Test Sets for the Current Class Model  
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
      
      x.test <- cnn_bin.ds_sample.set$test_set$x.test
      y.test <- cnn_bin.ds_sample.set$test_set$y.test
      test.files <- cnn_bin.ds_sample.set$test_set$x.files
      str(x.test)
      str(test.files)
      str(y.test)
      length(y.test)
    }
    
    ### Defining the Class Model Structure *************************************
    {  
      #> Now we define a CNN model with two 2D convolutional layers with max pooling, 
      #> and the 2nd layer with additonal dropout to prevent overfitting. 
      #> Then flatten the output and use two dense layers to connect to the categoires 
      #> of the image. [*]
      
      
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
    
    ##### Training Current  Model ******************************
    {  
      #> Now, we can train the model with our processed data. 
      #> Each cnn.bin.epochs's history can be saved to track the progress. 
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
            epochs = cnn.bin.epochs,
            batch_size = 50,
            validation_split = cnn.bin.vld_split,
            callbacks = cnn.lbl.callbacks
        )
      
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
           file = model.supporting_data.file_path)
      
      put_log("The (CNN) Binary Classifier Model for Class `%1` (%2) has been backed up to file:
%3",label.char, label,model.supporting_data.file_path)
      
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
  } 

  bin_model
  plot(cnn.1bl.train_history)
  
  #### Evaluate Pre-trained Model ************************************    
  {
    put_log("Evaluating the CNN-based Binary  Classifier model for '%1' character...", 
            label)
    eval.result <- bin_model |> evaluate(x.test, y.test)
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
    save(eval.result,
         preds,
         label,
         file = bin_model_eval.backup.path)
    put_log("CNN: the model evaluation data for label `%1` 
have been saved the the cache file:
%2", label.char, bin_model_eval.backup.path)
  }    
  
  ### Make Predictions & Analyse Results *********************************
  {    
    
    put_log("The current CNN Model evaluation result:
%1", capture.output(str(eval.result)))
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
    
    cnn_bin.conf.mx <- confusionMatrix(y.test.factor, predictions.factor)
    cnn_bin.conf.mx
    
    put_log("The confusion matrix for the binary model for recognizing 
the handwritten character '%1' is as follows:

%2", label, capture.output(cnn_bin.conf.mx))
    
    
    put_log("The CNN-based Binary  Classifier model evaluation task for '%1' character has been completed.",
            label)
  }
  
  put_end_date(start)
  
  list(saved_model.filepath = bin_model.file_path,
       lbl.data.cache.path = model.supporting_data.file_path,
       train_history = cnn.1bl.train_history,
       preds = preds,
       accuracy = accuracy,
       cnn_bin.conf.mx,
       # x.test = x.test,
       # y.test = y.test,
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

put_log("The CNN-based Binary Classifier Model evaluation job has been completed 
for each of the following handwritten characters:
%1", capture.output(as.character(y.labels)))

# stopCluster(cl)
# stopImplicitCluster()
# put_end_date(start)

bin_models.accuracies <- sapply(cnn_bin.models.dat, function(result){
  result$accuracy
})

# names(bin_models.accuracies) <- as.character(y.labels)
cnn.bin_models.accuracy <- data.frame(label = y.labels, accuracy = bin_models.accuracies) 
cnn.bin_models.accuracy

# put_log("Saving the CNN-based Binary Classifier Model evaluation results...")
# save(evaluation.results,
#      cnn.bin_models.accuracy,
#      file = cnn_bin.eval.backup.file_path)
# put_log("The CNN-based Binary Classifier model evaluation results have been saved to the following file:
# %1", cnn_bin.eval.backup.file_path)


put_log("The evaluation of the CNN-based Binary Classifier models results in the following accuracies:
%1", capture.output(cnn.bin_models.accuracy))
{
  # class  accuracy
  #' #     # 0.9995192
  #' $     $ 1.0000000
  #' &     & 0.9998077
  #' @     @ 0.9999342
  #' 0     0 0.9905732
  #' 1     1 0.9738435
  #' 2     2 0.9840534
  #' 3     3 0.9931875
  #' 4     4 0.9839958
  #' 5     5 0.9827479
  #' 6     6 0.9871399
  #' 7     7 0.9927580
  #' 8     8 0.9796366
  #' 9     9 0.9842771
  #' A     A 0.9622203
  #' B     B 0.9587421
  #' C     C 0.9828540
  #' D     D 0.9793681
  #' E     E 0.9849831
  #' F     F 0.9733219
  #' G     G 0.9022039
  #' H     H 0.9676555
  #' I     I 0.9670270
  #' J     J 0.9759390
  #' K     K 0.9734717
  #' L     L 0.9654734
  #' M     M 0.9909016
  #' N     N 0.9843641
  #' P     P 0.9810640
  #' Q     Q 0.9219745
  #' R     R 0.9760976
  #' S     S 0.9813778
  #' T     T 0.9856541
  #' U     U 0.9830638
  #' V     V 0.9803313
  #' W     W 0.9821060
  #' X     X 0.9677104
  #' Y     Y 0.9681717
  #' Z     Z 0.9825283
}

cnn.bin_models.accuracy |>
  data.plot("Average Accuracy", 
            xname = "label", 
            yname = "accuracy")

plot_bars.accuracy.by_class(y.labels,
                            cnn.bin_models.accuracy[, 2],
                            title.prefix = "CNN-based Binary")

# put_log("The CNN-based Binary Classifier Model evaluation job has been completed 
# for all the handwritten characters with the following results:
# %1", capture.output(str(evaluation.results)))

log_close()
