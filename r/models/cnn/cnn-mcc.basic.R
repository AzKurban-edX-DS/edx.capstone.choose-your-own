#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Multiclass Classifier Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Reference: Convolution Neural Network (CNN) 
# References:
# Building a Vision Inspection CNN for an Industrial Application
# https://towardsdatascience.com/building-a-vision-inspection-cnn-for-an-industrial-application-138936d7a34a/

# [*] Deep Learning Using R with keras (CNN) 
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html
## Prepare a Train Set for the Model Tuning ------------------------------------
open_logfile(".prepare-dataset-for-cnn-basic.model")
start <- put_start_date()
stopifnot(file.exists(train.img28x28mx.array.file_path))


put_log("Loading and splitting the Train 28x28 Image Data Array 
into a Default Train and Test Sets...")

split3d.list <- split.img28x28mx_array(train.img28x28mx.array.file_path,
                                       seed = N.classes,
                                       test_ratio = 0.9)

put_log("The Default Split Dataset object structure:
%1", capture.output(str(split3d.list)))

x3d.train_set <- split3d.list$train_set
# str(x3d.train_set)

put_log("The Train Set has been saved in the object `x3d.train_set`, 
which contains a training sample stored in the `x_train` variable having the following shape:
%1", capture.output(shape(x3d.train_set$x.train)))
# shape(132912, 28, 28)

rm(split3d.list)

x3d_train <- x3d.train_set$x.train

y.train.groups <- ds.get_classIDs.grouped(x3d_train)
y_train <- y.train.groups$classID


put_log("Reshaping the Train Set to make it compatible with the Convolutional Neural Network (CNN)...")
# Add channel into the dimension
x_train <- array_reshape(x3d_train, 
                         c(nrow(x3d.train_set$x.train), 
                           n.img_rows, 
                           n.img_cols, 
                           1))

put_log("The Train Set has been reshaped as follows:
%1", capture.output(shape(x_train)))
# shape(132912, 28, 28)

str(x_train)
dim(x_train)
shape(x_train)


rm(x3d.train_set,
   x3d_train)

stopifnot(sum(as.character(y_train) != rownames(x_train)) == 0)

# y_train <- as.array(as.integer(y_train) - 1)
# str(y_train)
# dim(y_train)
# 
# stopifnot(min(y_train) == 0)
# stopifnot(max(y_train) == 38)
stopifnot(length(y_train) == nrow(x_train))

put_log("The Train Set is balanced by the set of Classes:
%1", capture.output(print(y.train.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #         425
  #  2 $         425
  #  3 &         425
  #  4 @         425
  #  5 0         425
  #  6 1         425
  #  7 2         425
  #  8 3         425
  #  9 4         425
  # 10 5         425
  # 11 6         425
  # 12 7         425
  # 13 8         425
  # 14 9         425
  # 15 A         425
  # 16 B         425
  # 17 C         425
  # 18 D         425
  # 19 E         425
  # 20 F         425
  # 21 G         425
  # 22 H         425
  # 23 I         425
  # 24 J         425
  # 25 K         425
  # 26 L         425
  # 27 M         425
  # 28 N         425
  # 29 P         425
  # 30 Q         425
  # 31 R         425
  # 32 S         425
  # 33 T         425
  # 34 U         425
  # 35 V         425
  # 36 W         425
  # 37 X         425
  # 38 Y         425
  # 39 Z         425  
  invisible(NULL)
}

rm(y.train.groups)

str(x_train)
str(y_train)

dim(x_train)
dim(y_train)

y_train.cat <- to_categorical(y_train)
colnames(y_train.cat) <- Y.Labels

put_log("The Class Labels vector has been converted to a categorical matrix with the following dimensions:
%1", capture.output(dim(y_train.cat)))
# [1] 132912     39

# str(y_train.cat)
head(y_train.cat)
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0


## CNN MCC Model building --------------------------------------------------------------
open_logfile(".cnn-basic.model-building")
start <- put_start_date()

cnn_mcc.batch_size <- 128
cnn_mcc.epochs <- 100
cnn_mcc.vld_split <- 0.2

if(file.exists(cnn_mcc.model.file_path)) {
  put_log("Loading pre-trained CNN-Based Multiclass Classifier Model...")
  cnn_mcc.model <- load_model(cnn_mcc.model.file_path)
  put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_mcc.model.file_path)
  
  if(file.exists(cnn_mcc.train_history.file_path)){
    put_log("Loading the CNN-Based Multiclass Classifier Model Train History...")
    cnn_mcc.train_history <- readRDS(cnn_mcc.train_history.file_path)
    put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_mcc.train_history.file_path)
  } else {
    warning("The CNN-Based Multiclass Classifier Model backup does not exist:
", cnn_mcc.train_history.file_path)
  }
} else {
  #* *** Define a CNN-Based Multiclass Classification model structure *******
  {
    # Define a few parameters to be used in the CNN model
    
    #> Now we define a CNN model with two 2D convolutional layers with max pooling, 
    #> and the 2nd layer with additonal dropout to prevent overfitting. 
    #> Then flatten the output and use two dense layers to connect to the categoires 
    #> of the image. [*]
    
    # [*] 3.3.1 Define a CNN model structure
    cnn_mcc.inputs <- layer_input(shape = shape(28L, 28L, 1L))
    
    cnn_mcc.outputs <- cnn_mcc.inputs |>
      layer_conv_2d(filters = 32L,
                    kernel_size = c(3L, 3L), 
                    # strides = list(1L, 1L),
                    activation = "relu") |>
      # layer_max_pooling_2d() |>
      layer_max_pooling_2d(pool_size = c(2, 2)) |>
      # layer_dropout(rate = 0.25) |>
      layer_conv_2d(filters = 64L, 
                    kernel_size = c(3L, 3L),
                    # strides = list(1L, 1L),
                    activation = "relu") |>
      # layer_max_pooling_2d() |>
      layer_max_pooling_2d(pool_size = c(2, 2)) |>
      layer_dropout(rate = 0.25) |>
      layer_flatten() |>
      layer_dense(units = 128L, activation = "relu") |>
      layer_dropout(rate = 0.5) |>
      layer_dense(units = N.classes, activation = "softmax")
    
    
    cnn_mcc.model <- keras_model(cnn_mcc.inputs, cnn_mcc.outputs)

    # Similar to DNN model, we need to compile the defined CNN model. [*]
    
    # Compile model
    cnn_mcc.model |> compile(
      loss = loss_categorical_crossentropy,
      optimizer = keras3::optimizer_adamax(0.001),
      metrics = c('accuracy')
    )
    
    summary(cnn_mcc.model)
  }
  
  #' *** Training CNN-Based Muliclass Classifier Model **********************
  {
    #> Now, we can train the model with our processed data. 
    #> Each cnn_mcc.epochs's history can be saved to track the progress. 
    #> Please note, as we are not using GPU, it takes a few minutes to finish. 
    #> Please be patient while waiting for the results. 
    #> The training time can be significantly reduced if running on GPU. [*]
    dim(x_train)
    dim(y_train.cat)
    
    cnn_mcc.callbacks <- list(
      callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
      callback_model_checkpoint(filepath = cnn_mcc.checkpoint.file_path,
                                monitor = "val_loss",
                                mode = max,
                                save_best_only = TRUE,
                                verbose = 1)#,
      # callback_tensorboard(log_dir = data.cnn_mcc.tensorboard.logs.dir,
      #                      # write_graph = T,
      #                      write_images = T,
      #                      write_steps_per_second = T,
      #                      embeddings_freq = 1L)
    )
    
    # Runnign the TensorBoard tool
    #tensorboard(data.cnn_mcc.tensorboard.logs.dir)
    
    put_log("Training the CNN-based Multiclass Classifier (CNN MCC) Model...")
    start <- put_start_date()
    
    # Train model
    cnn_mcc.train_history <- cnn_mcc.model |> 
      fit(x_train, 
          y_train.cat,
          epochs = cnn_mcc.epochs,
          batch_size = cnn_mcc.batch_size,
          validation_split = cnn_mcc.vld_split,
          callbacks = cnn_mcc.callbacks
      )
    # acc: 0.8741
    
    put_log("Saving the pre-trained CNN MCC Model...")
    keras3::save_model(cnn_mcc.model,
                       filepath = cnn_mcc.model.file_path,
                       overwrite = TRUE)
    
    put_log("The CNN MCC has been trained 
and saved in the following file:
  %1", cnn_mcc.model.file_path)
    
    put_log("Saving the CNN MCC Model History...")
    saveRDS(cnn_mcc.train_history,
            file = cnn_mcc.train_history.file_path)
    
    put_log("The CNN MCC Model History has been trained 
and saved in the following file:
  %1", cnn_mcc.train_history.file_path)
  }
}


put_log("The CNN MCC Model has been created and trained:

%1", capture.output(cnn_mcc.model))


if(exists("cnn_mcc.train_history")){
  plot(cnn_mcc.train_history)
} 

put_end_date(start)

rm(x_train,
   y_train,
   y_train.cat,
   cnn_mcc.callbacks)

log_close()


