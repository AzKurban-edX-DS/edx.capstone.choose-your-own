## Explore imput datasets ------------------------------------------------------

str(img28x28bin.list$img.list$T$img.list)

lbl.img.list <- img28x28bin.list$img.list$T$img.list
str(lbl.img.list)

lbl.img.flat_ls <- lbl.img.list |> class_img.list2flatten_matrix(label)
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

cnn_models.ensemble.cache_file.path <- file.path(cnn.train.data.path,"cnn.lbl-models.ensemble.RData")
cnn_models.ensemble.cache_file.path

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

hwChar.CNN.binCls.models.backup.path <- file.path(cnn.train.data.path,"cnn.lbl-model.list.rds") 
hwChar.CNN.binCls.models.backup.path

cnn.lbl_model_file.base_name <- "cnn.lbl-model"

put_log("Building a set of CNN Binary Classifier Models for the following labels:
%1", capture.output(as.character(y.labels))) 

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
  
  lbl.model_cache.file_path <- file.path(data.cnn.binary.models.dir, 
                               str_flatten(c(cnn.lbl_model_file.base_name,
                                             label,
                                             as.character(label),
                                             "RData"),
                                           collapse = "."))
  
  lbl.model.file_path <- file.path(data.cnn.binary.models.dir, 
                                   str_flatten(c(cnn.lbl_model_file.base_name,
                                                 label,
                                                 as.character(label),
                                                 "keras"),
                                               collapse = "."))
  
  if (file.exists(lbl.model.file_path)) {
    put_log("CNN: loading (trained) labeled model data from cache file: 
%1", lbl.model.file_path)
    
    lbl.cnn_model <- load_model(lbl.model.file_path)
    str(lbl.cnn_model)
    load(lbl.model_cache.file_path)
    
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
    
    set.seed(as.integer(y.labels[y.labels == label]))
    
    lbl.ds.sample.set <- 
      cnn.binclass.sample_sets(x_cnn,
                               label)
    str(lbl.ds.sample.set)
    
    x_train <- lbl.ds.sample.set$x.train
    str(x_train)
    
    y_train <- lbl.ds.sample.set$y.train
    str(y_train)
    length(y_train)
    sum(y_train)
    
    x_test <- lbl.ds.sample.set$x.test
    y_test <- lbl.ds.sample.set$y.test
    
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
          y_train,
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
               overwrite = FALSE)
    
    put_log("The (CNN) Binary Classifier Model for label `%1` (%2) has been cached to file:
%3",as.character(label), label, lbl.model.file_path)
    
  }
  
  str(lbl.cnn_model)
  plot(cnn.1bl.train_history)
  
  list(model = lbl.cnn_model,
       saved_model.filepath = lbl.model.file_path,
       lbl.data.cache.path = lbl.model_cache.file_path,
       train_history = cnn.1bl.train_history,
       x_test = lbl.ds.sample.set$x.test,
       y_test = lbl.ds.sample.set$y.test,
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
      #        x_test               = lbl.ds.sample.set$x.test,
      #        y_test               = lbl.ds.sample.set$y.test,
      #        label                = label)
      
      x_test <- lbl.trained_model.list$x_test
      y_test <- lbl.trained_model.list$y_test

      put_log("Current model's object structure:
%1", capture.output(str(lbl.trained_model.list)))
      
      #exists("lbl.trained_model.list$train_history")
      plot(lbl.trained_model.list$train_history)

      put_log("Current model's object summary:
%1", capture.output(summary(lbl.trained_model.list)))
      
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
      lbl.eval.result <- lbl.cnn_model |> evaluate(x_test, y_test)
      put_log("Evaluation of the CNN-based Binary  Classifier model for '%1' character has been completed.",
              label)
      
      put_end_date(start)
      
      # model prediction
      put_log("Making predictions using the CNN-based Binary classifier model for the character '%1'...", 
              label)
      start <- put_start_date()
      preds <- lbl.cnn_model |> predict(x_test) 
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
    
    accuracy <- mean(predictions == y_test)
    put_log("CNN Model accuracy for label `%1`: %2", label, accuracy)
    # CNN Model accuracy for label `R`: 0.986499530817405
    
    y_test.factor <- as.factor(y_test)
    str(y_test.factor)
    
    predictions.factor <- factor(predictions, levels = levels(y_test.factor))
    str(predictions.factor)
    
    conf.mx <- confusionMatrix(y_test.factor, predictions.factor)
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

#### Ensemble Classifier for all labels ----------------------------------------
##### Open Log for Ensemble Classifier -----------------------------------------
open_logfile(".cnn-model.ensemble-classifier")
##### Build Ensemble Classifier ------------------------------------------------

if (file.exists(cnn_models.ensemble.cache_file.path)) {
  put_log("CNN: loading the Ensemble Classifier Results from cache file: 
%1", cnn_models.ensemble.cache_file.path)
  load(cnn_models.ensemble.cache_file.path)
  put_log("CNN: the Ensemble Classifier Results have been loaded from the cache file:
%1", cnn_models.ensemble.cache_file.path)
} else {
  
  if(!exists("evaluation.results")) {
    stopifnot(!file.exists(cnn_models.eval.cache_file.path))
    
    put_log("CNN: Loading the Labeled Binary Classifier Models evalution results from cache...") 
    load(cnn_models.eval.cache_file.path)
    put_log("CNN: the Labeled Binary Classifier Models evalution results 
have been loaded from the cache file:
%1", cnn_models.eval.cache_file.path)
    
  }
  
  preds.mx <- sapply(y.labels, function(label) {
    #p <- 
    evaluation.results[[label]]$preds[,1]
    #str(p)
  })
  
  
  class(preds.mx)
  dim(preds.mx)

  colnames(preds.mx) <- as.character(y.labels)
  str(preds.mx)
  head(preds.mx)
#               #            $            &            @            0            1
# [1,] 1.369949e-12 2.396904e-11 4.014774e-10 3.622346e-09 1.100269e-06 3.533862e-05
# [2,] 2.899649e-13 4.355252e-11 4.055812e-14 1.342694e-10 8.238061e-07 1.023313e-06
# [3,] 1.046244e-07 6.048190e-11 2.396136e-08 9.999999e-01 3.843675e-13 2.751844e-09
# [4,] 9.592652e-08 1.500224e-05 9.999545e-01 1.799757e-07 2.254854e-12 2.517688e-09
# [5,] 6.245161e-10 4.554279e-12 1.529452e-06 4.037680e-11 2.591312e-10 1.986288e-03
# [6,] 6.703245e-15 7.939808e-09 7.806473e-10 9.999957e-01 1.411460e-13 2.716332e-08
#               2            3            4            5            6            7
# [1,] 1.274050e-07 1.976297e-08 2.871119e-07 9.117506e-01 8.310264e-01 1.663653e-12
# [2,] 1.036329e-04 5.842682e-08 7.924597e-06 2.290502e-03 9.897284e-01 2.419553e-16
# [3,] 2.698742e-09 4.878437e-09 2.208877e-07 1.066727e-06 1.853692e-09 9.732599e-10
# [4,] 2.360041e-07 3.376738e-05 4.101317e-08 6.470874e-04 5.833106e-08 2.433209e-13
# [5,] 1.783007e-03 6.626777e-07 7.431519e-07 2.365470e-05 1.260650e-07 8.708229e-05
# [6,] 3.107357e-10 2.704447e-08 7.753220e-05 3.603433e-06 3.583733e-05 1.783171e-09
#               8            9            A            B            C            D
# [1,] 5.372741e-05 1.092196e-07 5.041434e-05 2.950370e-04 4.660272e-07 1.776997e-05
# [2,] 6.252207e-04 3.709382e-07 8.188316e-03 7.029655e-04 2.369297e-05 2.283386e-05
# [3,] 7.717297e-07 1.682601e-07 5.947574e-06 2.228543e-05 2.070900e-07 2.059591e-08
# [4,] 1.309585e-06 2.195027e-03 1.557932e-05 4.677896e-06 7.163868e-07 7.496845e-08
# [5,] 4.989845e-03 3.979222e-07 8.075874e-02 5.097317e-04 9.486013e-08 3.500879e-03
# [6,] 1.676496e-08 6.705029e-05 2.840243e-05 3.444430e-05 5.380749e-07 1.841350e-10
#               E            F            G            H            I            J
# [1,] 8.316667e-03 3.006915e-04 1.442758e-01 1.378821e-04 3.178597e-07 4.436253e-06
# [2,] 1.033750e-02 8.659600e-03 1.065786e-01 3.527990e-05 9.302857e-08 2.817498e-10
# [3,] 4.010222e-09 9.375635e-05 5.761859e-03 1.861690e-03 1.403905e-09 7.124655e-09
# [4,] 1.788994e-08 1.207731e-06 1.175228e-03 5.415139e-06 4.145133e-10 2.589490e-09
# [5,] 4.289760e-10 1.531458e-03 6.374015e-05 5.685989e-02 8.214690e-05 1.906513e-04
# [6,] 1.403405e-08 3.572582e-05 5.061809e-03 6.416230e-03 4.529754e-08 2.157732e-12

  bin_preds.mx <- (preds.mx > 0.5) |> 
    as.integer() |> 
    matrix(nrow = nrow(preds.mx))
  
  colnames(bin_preds.mx) <- as.character(y.labels)
  
  class(bin_preds.mx)
  dim(bin_preds.mx)
  #> [1] 817379     39
  
  head(bin_preds.mx)
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  
  
  df.model.acc_ordered <- df.models.acc |> arrange(accuracy)
  head(df.model.acc_ordered)
  # label  accuracy
  # T     T 0.9627761
  # 1     1 0.9708605
  # L     L 0.9711284
  # 5     5 0.9797157
  # I     I 0.9804546
  # U     U 0.9804766 
  
  lbl_T.idx <- df.model.acc_ordered$label == "T"
  lbl_T.acc <- df.model.acc_ordered[lbl_T.idx,]$accuracy
  lbl_T.acc
  # 0.9627761
  
  y.T_test <- (y_test == "T") |> as.integer()
  str(y.T_test)
 # int [1:817379] 0 0 0 0 0 0 0 0 0 0 ...

  bin.T_pred.mx <- bin_preds.mx[,"T"]
  str(bin.T_pred.mx)
  #  int [1:817379] 0 0 0 0 0 0 0 0 0 0 ..

  sum(bin.T_pred.mx)
  # 0
  
  sum(y.T_test)
  # Accuracy for binary classifier:
  sum(bin.T_pred.mx == y.T_test)
  # 786953
  
  mean(bin.T_pred.mx == y.T_test)
  #> [1] 0.9627761
  
  y.T_test.factor <- as.factor(y.T_test)
  str(y.T_test.factor)
  
  bin.T_pred.mx.factor <- factor(bin.T_pred.mx, levels = levels(y.T_test.factor))
  str(bin.T_pred.mx.factor)
  
  conf.mx <- confusionMatrix(y.T_test.factor, bin.T_pred.mx.factor)
  conf.mx
  # Confusion Matrix and Statistics
  # 
#     Reference
  # Prediction      0      1
  # 0          786953      0
  # 1          30426      0
  # 
  # Accuracy : 0.9628          
  # 95% CI : (0.9624, 0.9632)
  # No Information Rate : 1               
  # P-Value [Acc > NIR] : 1               
  # 
  # Kappa : 0               
  # 
  # Mcnemar's Test P-Value : <2e-16          
  #                                         
  #           Sensitivity : 0.9628          
  #           Specificity :     NA          
  #        Pos Pred Value :     NA          
  #        Neg Pred Value :     NA          
  #            Prevalence : 1.0000          
  #        Detection Rate : 0.9628          
  #  Detection Prevalence : 0.9628          
  #     Balanced Accuracy :     NA          
  #                                         
  #      'Positive' Class : 0             
  
  lbl_1.idx <- df.model.acc_ordered$label == "1"
  lbl_1.acc <- df.model.acc_ordered[lbl_1.idx,]$accuracy
  lbl_1.acc
  # 0.9708605
  
  lbl_L.idx <- df.model.acc_ordered$label == "L"
  lbl_L.acc <- df.model.acc_ordered[lbl_L.idx,]$accuracy
  lbl_L.acc
  # 0.9711284
  
  names(evaluation.results) <- as.character(y.labels)
  
  avg.accuracy = mean(lbl_models.accuracies)
  avg.accuracy
  # 0.9888955
  
  head(preds.mx[,"T"])
  
    
  # rMaxs <- rowMaxs(preds.mx)
  # str(rMaxs)
  # head(rMaxs, 50)
  
  preds.optimistic <- apply(preds.mx, 1, function(r) {
    c(label = as.character(y.labels)[which.max(r)], 
      P = max(r))
  }) |> t()
  
  class(preds.optimistic)
  # "matrix" "array"
  dim(preds.optimistic)
  # 817379      2
  str(preds.optimistic)
  
  head(preds.optimistic)
#       label P                   
# [1,] "5"   "0.911750555038452" 
# [2,] "6"   "0.989728391170502" 
# [3,] "@"   "0.99999988079071"  
# [4,] "&"   "0.999954521656036" 
# [5,] "A"   "0.0807587429881096"
# [6,] "@"   "0.999995708465576" 
  
  length(y_test)
  # 817379
  
  head(y_test)
  # [1] 5 6 @ & X @
  # 39 Levels: # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T ... Z

  acc.optimistic <- mean(y_test == preds.optimistic[,1]) 
  acc.optimistic
  #> 0.7836793
  
  y.labels.ext <- y.labels
  levels(y.labels.ext) <- c(levels(y.labels), "NA")
  y.labels.ext[40] <- "NA"
  y.labels.ext
  
  
  preds.norm <- apply(preds.mx, 1, function(r) {
   
    r.max = max(r)
    label <- as.character(y.labels)[which(r == r.max)]
      
    c(label = ifelse(r.max > 0.5, label, "NA"), 
      P = r.max)
  }) |> t()
  
  class(preds.norm)
  # "matrix" "array"
  dim(preds.norm)
  # 817379      2
  str(preds.norm)
  
  head(preds.norm)
#        label P                   
# [1,] "5"   "0.911750555038452" 
# [2,] "6"   "0.989728391170502" 
# [3,] "@"   "0.99999988079071"  
# [4,] "&"   "0.999954521656036" 
# [5,] "NA"  "0.0807587429881096"
# [6,] "@"   "0.999995708465576" 

  acc.norm <- mean(y_test == preds.norm[,1]) 
  acc.norm
  # [1] 0.7429589
 
  
  
  preds.pessimistic <- apply(preds.mx, 1, function(r) {
    
    r.max = max(r)
    label <- as.character(y.labels)[which(r == r.max)]
    
    c(label = ifelse(r.max > 0.75, label, "NA"), 
      P = r.max)
  }) |> t()
  
  class(preds.pessimistic)
  # "matrix" "array"
  dim(preds.pessimistic)
  # 817379      2
  str(preds.pessimistic)
  
  head(preds.pessimistic)
  #        label P                   
  # [1,] "5"   "0.911750555038452" 
  # [2,] "6"   "0.989728391170502" 
  # [3,] "@"   "0.99999988079071"  
  # [4,] "&"   "0.999954521656036" 
  # [5,] "NA"  "0.0807587429881096"
  # [6,] "@"   "0.999995708465576" 
  
  acc.pessimistic <- mean(y_test == preds.pessimistic[,1]) 
  acc.pessimistic
  # [1] 0.6619133
  
  
  
      
  put_log("CNN: Caching the Ensemble Classifier Results...")
  save(cnn.ensemble,
       y_test,
       file = cnn_models.ensemble.cache_file.path)
  put_log("CNN: the Ensemble Classifier Results have been saved to the cache file:
%1", cnn_models.ensemble.cache_file.path)
  
  
}

#### Close Log -----------------------------------------------------------------
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
