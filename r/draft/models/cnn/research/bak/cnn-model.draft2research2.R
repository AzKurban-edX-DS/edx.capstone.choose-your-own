str(img28x28bin.list$img.list$T$img.list)

lbl.img.list <- img28x28bin.list$img.list$T$img.list
str(lbl.img.list)

lbl.img.flat_ls <- lbl.img.list |> img_mx.list2flatten_matrix(label)
str(lbl.img.flat_ls)

image(lbl.img.flat_ls[1:400,])

#### Init CNN Directories -------------------------------------------
# Reference: https://tensorflow.rstudio.com/guides/keras/basics.html#callbacks

data.cnn.binary.models.dir <- file.path(cnn.train.data.path, "lbl-models")

if(!dir.exists(data.cnn.binary.models.dir))
  dir.create(data.cnn.binary.models.dir)

cnn.eval.cache.path <- file.path(cnn.train.data.path, "evaluation")

if(!dir.exists(cnn.eval.cache.path))
  dir.create(cnn.eval.cache.path)

cnn.callbacks.path <- file.path(cnn.train.data.path, "callbacks")

if(!dir.exists(cnn.callbacks.path))
  dir.create(cnn.callbacks.path)

cnn.callbacks.tensorboard.path <- file.path(cnn.callbacks.path, "tensorboard")

if(!dir.exists(cnn.callbacks.tensorboard.path))
  dir.create(cnn.callbacks.tensorboard.path)

cnn.callbacks.tb_logs.path <- file.path(cnn.callbacks.tensorboard.path, "logs")

if(!dir.exists(cnn.callbacks.tb_logs.path))
  dir.create(cnn.callbacks.tb_logs.path)


cnn.callbacks.checkpoints.path <- file.path(cnn.callbacks.path, "checkpoints")

if(!dir.exists(cnn.callbacks.checkpoints.path))
  dir.create(cnn.callbacks.checkpoints.path)

##### Init CNN Datasets ------------------------------------------------------------
x_train <- x_cnn.1bl.train
str(x_train)

y_train <- y_cnn.1bl.train
str(y_train)
length(y_train)

x_test <- x_cnn.9.test
dim(x_test)

y_test <- y_cnn.9.test
str(y_test)
length(y_test)

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

##### Define a CNN model structure ***

cache_file.path <- file.path(cnn.train.data.path,"cnn.lbl-model.list.RData") 
                             
if (file.exists(cache_file.path)) {
  put_log("CNN: loading trained model list data from cache file: 
%1", cache_file.path)
  load(cache_file.path)
  put_log("CNN: the trained model list data have been loaded from the cache file:
%1", cache_file.path)
} else {
  cnn.model_cache.base_name <- "cnn.lbl-model"
  
  put_log("Building a set of CNN Binary Classifier Models for the following labels:
%1", capture.output(as.character(y.labels))) 

  cnn.hw_char.models <- lapply(y.labels, function(label) {
    #> Now we define a CNN model with two 2D convolutional layers with max pooling, 
    #> and the 2nd layer with additonal dropout to prevent overfitting. 
    #> Then flatten the output and use two dense layers to connect to the categoires 
    #> of the image. [*]
    
    put_log("Building model for label `%1` (%2)...", as.character(label), label)
    
    cache_file.path <- file.path(data.cnn.binary.models.dir, 
                                 str_flatten(c(cnn.model_cache.base_name,
                                               label,
                                               as.character(label),
                                               "RData"),
                                             collapse = "."))
    
    lbl.model.file_path <- file.path(data.cnn.binary.models.dir, 
                                 str_flatten(c(cnn.model_cache.base_name,
                                               label,
                                               as.character(label),
                                               "keras"),
                                             collapse = "."))
    
    if (file.exists(lbl.model.file_path)) {
      put_log("CNN: loading (trained) labeled model data from cache file: 
%1", lbl.model.file_path)
      # load(cache_file.path)
      
      put_log("CNN: the (trained) labeled model data has been loaded from the cache file:
%1", lbl.model.file_path)
    } else {
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
        file.path(data.cnn.binary.models.dir,
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

      y <- as.integer(y_train == label)
      str(y)
      length(y)
      sum(y)
      
      put_log("Training the CNN Model...")
      start <- put_start_date()
      
      #str(x_train)
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
        fit(x_train, 
            y,
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
      plot(cnn.1bl.train_history)
      
      put_log("Caching the (CNN) Binary Classifier Model for label `%1` (%2)...", 
              as.character(label), 
              label)

      save(lbl.model.file_path,
           cnn.1bl.train_history,
           file = cache_file.path)
      
      put_log("The (CNN) Binary Classifier Model for label `%1` (%2) has been cached to file:
%3",as.character(label), label,cache_file.path)
    
      put_log("Saving the (CNN) Binary Classifier Model for label `%1`to file...", 
              label)
      
      save_model(lbl.cnn_model,
                 filepath = lbl.model.file_path,
                 overwrite = FALSE)
      
      put_log("The (CNN) Binary Classifier Model for label `%1` (%2) has been cached to file:
%3",as.character(label), label, lbl.model.file_path)
      
    }
    
    list(model = lbl.cnn_model,
         saved_model.filepath = lbl.model.file_path,
         train_history = cnn.1bl.train_history,
         label = label)
  })

  names(cnn.hw_char.models) = as.character(y.labels)
  
  put_log("Saving the set of trained (CNN) Binary Classifier Models to file...") 

  save(cnn.hw_char.models,
       file = cache_file.path)

  put_log("The set of trained (CNN) Binary Classifier Models have been saved to file:
%1", cache_file.path)
}

### Close Log ------------------------------------------------------------------
log_close()
#### Open log: Evaluate CNN Model -------------------------------------------------
open_logfile(".evaluate-cnn-model")
#### Evaluating CNN Model ----------------------------------------------
put_log("CNN Models Evaluation started for entire set of labels...")
start_eval_time <- put_start_date()

cache_file.path <- file.path(cnn.train.data.path,"cnn.lbl-models.evaluation.RData") 

if (file.exists(cache_file.path)) {
  put_log("CNN: loading models set evaluation data from cache file: 
%1", cache_file.path)
  load(cache_file.path)
  put_log("CNN: the models set evaluation data have been loaded from the cache file:
%1", cache_file.path)
} else {
  cnn.eval_cache.base_name <- "cnn.lbl-model.eval"
  
  evaluation.results <- lapply(y.labels, function(label) {
    put_log("Processing model evaluation for label `%1`...",label)
    start <- put_start_date()

    y <- as.integer(y_test == label)
    str(y)
    length(y)
    sum(y)

    cache_file.path <- file.path(cnn.eval.cache.path, 
                                 str_flatten(c(cnn.eval_cache.base_name,
                                               label,
                                               as.character(label),
                                               "RData"),
                                             collapse = "."))
    
    put_log("Cache file path: %1", cache_file.path)
    
    if (file.exists(cache_file.path)) {
      put_log("CNN: loading model evaluation data for label `%1` from cache...", label)
      load(cache_file.path)
      put_log("CNN: the model evaluation data for label `%1` have been loaded from the cache file:
%2", label, cache_file.path)
    } else {
      
      lbl.cnn_model.ls <- cnn.hw_char.models[[label]] 
      
      put_log("Current model's object structure:
%1", capture.output(str(lbl.cnn_model.ls)))
      
      put_log("Current model's object summary:
%1", capture.output(summary(lbl.cnn_model.ls)))
      
      lbl.cnn_model <- lbl.cnn_model.ls$model
      
      put_log("Evaluating current CNN Model...")
      start <- put_start_date()
      lbl.eval.result <- lbl.cnn_model |> evaluate(x_test, y)
      put_log("The current CNN Model evaluation result:
%1", capture.output(str(lbl.eval.result)))
      # CNN Model evaluation result:
      #   List of 2
      # $ accuracy: num 0.963
      # $ loss    : num 0.752
      put_end_date(start)
      
      # model prediction
      put_log("CNN Model for label `%1`: constructing predictions...", label)
      start <- put_start_date()
      preds <- lbl.cnn_model |> predict(x_test) 
      put_log("CNN Model for label `%1`: predictions have been constructed.", label)
      put_end_date(start)
      # Time difference of 1.502232 mins
      str(preds)
      head(preds)
      # max(preds)

      # predictions <- (as.vector(preds) > 0.5) |> as.integer()
      # predictions <- preds %>% `>` (0.5) |> as.integer()
      predictions <- cnn.binclass.get_prediction_values(preds)
      str(predictions)
      sum(predictions)
      
      accuracy <- mean(predictions == y)
      put_log("CNN Model accuracy for label `%1`: %2", label, accuracy)
      # CNN Model accuracy for label `R`: 0.986499530817405

      put_log("CNN: Saving model evaluation data for label `%1` to cache...", label)
      save(y,
           preds,
           label,
           accuracy,
           file = cache_file.path)
      put_log("CNN: the model evaluation data for label `%1` have been saved the the cache file:
%2", label, cache_file.path)
    }
    
    put_log("Model evaluation process for label `%1` has been completed.",label)
    put_end_date(start)
    
    list(y = y,
         preds = preds,
         label = label,
         accuracy = accuracy)
  })

  names(evaluation.results) <- as.character(y.labels)
  
  save(evaluation.results,
       y_test,
       file = cache_file.path)
}

put_log("CNN Models Evaluation has been completed with the following results:
%1", capture.output(str(evaluation.results)))

put_end_date(start_eval_time)
### Close Log ------------------------------------------------------------------
log_close()



#### Ensemble Classifier for all labels ----------------------------------------
open_logfile(".cnn-model.ensemble-classifier")

cnn.ensemble <- lapply(y.labels, function(label) {
  eval.resut <- evaluation.results[[label]]$preds
})


#### Close Log ------------------------------------------------------------------
log_close()
#----------------------------------
err.idx <- which(cnn.prediction.values != as.integer(y_cnn.test))
length(err.idx)
# 94951
err.head.idx <- head(err.idx)



err.pred.values <- cnn.prediction.values[err.idx]
head(err.pred.values)

err.teslbl.values <- y_cnn.test[err.idx]
head(err.teslbl.values)

err.head.img <- x_test[err.head.idx,,,1]
dim(err.head.img)

par(mfrow = c(6, 1))
for(i in err.head.idx) {
  char.image(x_test[i,,,1])
}
par(mfrow = c(1,1))

#> [*] Reference: https://databricks-prod-cloudfronlbl.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/lateslbl.html
