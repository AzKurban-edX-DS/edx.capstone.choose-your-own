#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Basic CNN Multiclass Classifier (MCC) Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Reference: Convolution Neural Network (CNN) 
# References:
# Building a Vision Inspection CNN for an Industrial Application
# https://towardsdatascience.com/building-a-vision-inspection-cnn-for-an-industrial-application-138936d7a34a/

# [*] Deep Learning Using R with keras (CNN) 
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

## Setup -----------------------------------------------------------------------

open_logfile(".cnnb-mcc.model-building")

start <- put_start_date()
stopifnot(file.exists(train.img28x28mx.array.file_path))

## Prepare a Data Set for the Model Training -----------------------------------

put_log("Loading the Training Set of 28x28x1-shape image data...")

train_set <- load28x28x1.train_set(ds28x28.split.train_0.8.backup.file)
put_log("The Training Set of 28x28x1-shape image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)

x_train <- train_set$x

y_train <- train_set$class_groups$classID

stopifnot(sum(as.character(y_train) != rownames(x_train)) == 0,
          length(y_train) == nrow(x_train))

### Size of the Training Set by Class ------------------------------------------
put_log("The Training Set is balanced by the set of Classes:
%1", capture.output(print(train_set$class_groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
#  1 #        3407
#  2 $        3407
#  3 &        3407
#  4 @        3407
#  5 0        3407
#  6 1        3407
#  7 2        3407
#  8 3        3407
#  9 4        3407
# 10 5        3407
# 11 6        3407
# 12 7        3407
# 13 8        3407
# 14 9        3407
# 15 A        3407
# 16 B        3407
# 17 C        3407
# 18 D        3407
# 19 E        3407
# 20 F        3407
# 21 G        3407
# 22 H        3407
# 23 I        3407
# 24 J        3407
# 25 K        3407
# 26 L        3407
# 27 M        3407
# 28 N        3407
# 29 P        3407
# 30 Q        3407
# 31 R        3407
# 32 S        3407
# 33 T        3407
# 34 U        3407
# 35 V        3407
# 36 W        3407
# 37 X        3407
# 38 Y        3407
# 39 Z        3407
  invisible(NULL)
}

rm(train_set)

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


## Init File Paths -----------------------------------------------------------

data.cnn_mcc.basic.tensorboard.dir <- file.path(data.cnn_mcc.basic.dir, "tensorboard")

if(!dir.exists(data.cnn_mcc.basic.tensorboard.dir))
  dir.create(data.cnn_mcc.basic.tensorboard.dir)

data.cnn_mcc.basic.tensorboard.logs.dir <- file.path(data.cnn_mcc.basic.tensorboard.dir, "logs")

if(!dir.exists(data.cnn_mcc.basic.tensorboard.logs.dir))
  dir.create(data.cnn_mcc.basic.tensorboard.logs.dir)

data.cnn_mcc.basic.checkpoints.dir <- file.path(data.cnn_mcc.basic.dir, "checkpoints")

if(!dir.exists(data.cnn_mcc.basic.checkpoints.dir))
  dir.create(data.cnn_mcc.basic.checkpoints.dir)

cnn_mcc.basic.checkpoint.file_path <- 
  file.path(data.cnn_mcc.basic.checkpoints.dir, 
            "{epoch:02d}-{val_loss:.2f}.keras")

cnn_mcc.basic.plot_img.file <- file.path(cnn_mcc.basic.plots.dat.dir,
                                         "cnn-mcc.basic-model.png")

## Basic CNN MCC Model building --------------------------------------------------------------
start <- put_start_date()

cnn_mcc.basic.batch_size <- 128
cnn_mcc.basic.epochs <- 100
cnn_mcc.basic.vld_split <- 0.2

if(file.exists(cnn_mcc.basic.file_path)) {
  put_log("Loading pre-trained CNN-Based Multiclass Classifier Model...")
  cnn_mcc.basic <- keras3::load_model(cnn_mcc.basic.file_path)
  put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_mcc.basic.file_path)
  
  if(file.exists(cnn_mcc.basic.train_history.file_path)){
    put_log("Loading the CNN-Based Multiclass Classifier Model Train History...")
    cnn_mcc.basic.train_history <- readRDS(cnn_mcc.basic.train_history.file_path)
    put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_mcc.basic.train_history.file_path)
  } else {
    warning("The CNN-Based Multiclass Classifier Model backup does not exist:
", cnn_mcc.basic.train_history.file_path)
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
    cnn_mcc.basic.inputs <- layer_input(shape = shape(28L, 28L, 1L))
    
    cnn_mcc.basic.outputs <- cnn_mcc.basic.inputs |>
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
      layer_max_pooling_2d(pool_size = c(2L, 2L)) |>
      layer_dropout(rate = 0.25) |>
      layer_flatten() |>
      layer_dense(units = 128L, activation = "relu") |>
      layer_dropout(rate = 0.5) |>
      layer_dense(units = N.classes, activation = "softmax")
    
    
    cnn_mcc.basic <- keras_model(cnn_mcc.basic.inputs, cnn_mcc.basic.outputs)

    # Similar to DNN model, we need to compile the defined CNN model. [*]
    
    # Compile model
    cnn_mcc.basic |> compile(
      loss = loss_categorical_crossentropy,
      optimizer = keras3::optimizer_adamax(0.001),
      metrics = c('accuracy')
    )
    
    summary(cnn_mcc.basic)
    
    cnn_mcc.basic |> plot_keras_model(to_file = cnn_mcc.basic.plot_img.file,
                                            show_shapes = T)
  }
  
  #' *** Training CNN-Based Muliclass Classifier Model **********************
  {
    #> Now, we can train the model with our processed data. 
    #> Each cnn_mcc.basic.epochs's history can be saved to track the progress. 
    #> Please note, as we are not using GPU, it takes a few minutes to finish. 
    #> Please be patient while waiting for the results. 
    #> The training time can be significantly reduced if running on GPU. [*]
    dim(x_train)
    dim(y_train.cat)
    
    cnn_mcc.basic.callbacks <- list(
      callback_early_stopping(patience = 3, monitor = 'val_accuracy'),
      callback_model_checkpoint(filepath = cnn_mcc.basic.checkpoint.file_path,
                                monitor = "val_loss",
                                mode = max,
                                save_best_only = TRUE,
                                verbose = 1)#,
      # callback_tensorboard(log_dir = data.cnn_mcc.basic.tensorboard.logs.dir,
      #                      # write_graph = T,
      #                      write_images = T,
      #                      write_steps_per_second = T,
      #                      embeddings_freq = 1L)
    )
    
    # Runnign the TensorBoard tool
    #tensorboard(data.cnn_mcc.basic.tensorboard.logs.dir)
    
    put_log("Training the CNN-based Multiclass Classifier (CNN MCC) Model...")
    start <- put_start_date()
    
    # Train model
    cnn_mcc.basic.train_history <- cnn_mcc.basic |> 
      fit(x_train, 
          y_train.cat,
          epochs = cnn_mcc.basic.epochs,
          batch_size = cnn_mcc.basic.batch_size,
          validation_split = cnn_mcc.basic.vld_split,
          callbacks = cnn_mcc.basic.callbacks
      )
    # acc: 0.8741
    
    put_log("Saving the pre-trained CNN MCC Model...")
    keras3::save_model(cnn_mcc.basic,
                       filepath = cnn_mcc.basic.file_path,
                       overwrite = TRUE)
    
    put_log("The CNN MCC has been trained 
and saved in the following file:
  %1", cnn_mcc.basic.file_path)
    
    put_log("Saving the CNN MCC Model History...")
    saveRDS(cnn_mcc.basic.train_history,
            file = cnn_mcc.basic.train_history.file_path)
    
    put_log("The CNN MCC Model History has been trained 
and saved in the following file:
  %1", cnn_mcc.basic.train_history.file_path)
  }
}

put_log("The CNN MCC Model has been created and trained:
%1", capture.output(cnn_mcc.basic))

if(exists("cnn_mcc.basic.train_history")){
  plot(cnn_mcc.basic.train_history)
} 

put_end_date(start)

rm(x_train,
   y_train,
   y_train.cat,
   cnn_mcc.basic.callbacks)


log_close()
