#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Multiclass Classifier Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Reference: Convolution Neural Network (CNN) 
# Reference:
# Deep Learning Using R with keras (CNN)
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

## Preparing Training Data ---------------------------------------------------------

open_logfile(".train.cnn_multiclass-classifier.model")

put_log("Preparing Training Data...")
start <- put_start_date()

put_log("The Train Set object (`x3d.train_set`) hase the following structure:
%1", capture.output(str(x3d.train_set)))

class.groups <- ds.get_classIDs.grouped(x3d.train_set$x.train)
y.train <- class.groups$classID
length(y.train)
#> [1] 132912

y.train.cat <- to_categorical(y.train)
colnames(y.train.cat) <- y.labels

put_log("The Class Labels vector has been converted to a categorical matrix with the following dimensions:
%1", capture.output(dim(y.train.cat)))
# [1] 132912     39

# str(y.train.cat)
head(y.train.cat)
#      # $ & @ 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N P Q R S T U V W X Y Z
# [1,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [2,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [3,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
# [4,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
# [5,] 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# [6,] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0

put_log("Reshaping the Train Set to make it compatible with the Convolutional Neural Network (CNN)...")
# Add channel into the dimension
x.train <- array_reshape(x3d.train_set$x.train, 
                             c(nrow(x3d.train_set$x.train), 
                               n.img_rows, 
                               n.img_cols, 
                               1))

put_log("The Train Set has been reshaped as follows:
%1", capture.output(shape(x.train)))
# shape(132912, 28, 28)

### class Identifies: Quick Analysis -------------------------------------------

y.train.chars <- class.groups$groupByClass
#str(y.train.chars)

char_n.max <- max(y.train.chars$n)
# 3408
char_n.max == min(y.train.chars$n)
# TRUE

put_log("The number of rows for each *Character Class* to be recognized in the Train Set is as follows:
%1", capture.output(print(y.train.chars, n = nrow(y.train.chars))))
{
# A tibble: 39 × 2
#    classID     n
#    <fct>   <int>
#  1 #        3408
#  2 $        3408
#  3 &        3408
#  4 @        3408
#  5 0        3408
#  6 1        3408
#  7 2        3408
#  8 3        3408
#  9 4        3408
# 10 5        3408
# 11 6        3408
# 12 7        3408
# 13 8        3408
# 14 9        3408
# 15 A        3408
# 16 B        3408
# 17 C        3408
# 18 D        3408
# 19 E        3408
# 20 F        3408
# 21 G        3408
# 22 H        3408
# 23 I        3408
# 24 J        3408
# 25 K        3408
# 26 L        3408
# 27 M        3408
# 28 N        3408
# 29 P        3408
# 30 Q        3408
# 31 R        3408
# 32 S        3408
# 33 T        3408
# 34 U        3408
# 35 V        3408
# 36 W        3408
# 37 X        3408
# 38 Y        3408
# 39 Z        3408
}

## Model building --------------------------------------------------------------

n.output <- N.classes
batch_size <- 128
num_classes <- 39
epochs <- 100
vld_split <- 0.2

if(file.exists(cnn_multiclass.model.file_path)) {
  put_log("Loading pre-trained CNN-Based Multiclass Classifier Model...")
  cnn_multiclass.model <- load_model(cnn_multiclass.model.file_path)
  put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_multiclass.model.file_path)
  
  if(file.exists(cnn_multiclass.train_history.file_path)){
    put_log("Loading the CNN-Based Multiclass Classifier Model Train History...")
    cnn_multiclass.train_history <- readRDS(cnn_multiclass.train_history.file_path)
    put_log("The CNN-Based Multiclass Classifier Model has been loaded from the backup file:
%1", cnn_multiclass.train_history.file_path)
  } else {
    warning("The CNN-Based Multiclass Classifier Model backup does not exist:
", cnn_multiclass.train_history.file_path)
  }
} else {
  ##### Define a CNN-Based Multiclass Classification model structure ***********
  {
    # Define a few parameters to be used in the CNN model
    
    #> Now we define a CNN model with two 2D convolutional layers with max pooling, 
    #> and the 2nd layer with additonal dropout to prevent overfitting. 
    #> Then flatten the output and use two dense layers to connect to the categoires 
    #> of the image. [*]
    
    cnn_multiclass.model <- keras_model_sequential(shape(28L, 28L, 1L)) |>
      layer_conv_2d(filters = 8L,
                    kernel_size = 5, 
                    strides = 1,
                    activation = "relu") |>
      layer_max_pooling_2d(strides = c(2, 2)) |>
      layer_dropout(rate = 0.25) |>
      layer_conv_2d(filters = 16L, 
                    kernel_size = 5,
                    strides = 1,
                    activation = "relu") |>
      layer_max_pooling_2d(strides = c(2, 2)) |>
      layer_flatten() |>
      layer_dense(units = 128, activation = "relu") |>
      layer_dropout(rate = 0.3) |>
      layer_dense(units = n.output, activation = "softmax")
    
    summary(cnn_multiclass.model)
    # plot(cnn_multiclass.model)
    
    # Similar to DNN model, we need to compile the defined CNN model. [*]
    
    # Compile model
    cnn_multiclass.model |> compile(
      loss = loss_categorical_crossentropy,
      optimizer = keras3::optimizer_adamax(0.001),
      metrics = c('accuracy')
    )
    
    summary(cnn_multiclass.model)
  }
  
  #### Training CNN-Based Muliclass Classifier Model ***************************
  {
    #> Now, we can train the model with our processed data. 
    #> Each epochs's history can be saved to track the progress. 
    #> Please note, as we are not using GPU, it takes a few minutes to finish. 
    #> Please be patient while waiting for the results. 
    #> The training time can be significantly reduced if running on GPU. [*]
    dim(x.train)
    dim(y.train.cat)
    
    cnn_multiclass.callbacks <- list(
      callback_model_checkpoint(filepath = cnn_multiclass.checkpoint.file_path,
                                monitor = "val_accuracy",
                                mode = max,
                                # save_best_only = TRUE,
                                verbose = 1)
    )
    
    put_log("Training the CNN-based Multiclass Classifier (CNN MCC) Model...")
    start <- put_start_date()
    
    # Train model
    cnn_multiclass.train_history <- cnn_multiclass.model |> 
      fit(x.train, 
          y.train.cat,
          epochs = epochs,
          batch_size = batch_size,
          validation_split = vld_split,
          callbacks = cnn_multiclass.callbacks
      )
    # acc: 0.8741
    
    put_log("Saving the pre-trained CNN MCC Model...")
    save_model(cnn_multiclass.model,
               filepath = cnn_multiclass.model.file_path,
               overwrite = TRUE)
    
    put_log("The CNN MCC has been trained 
and saved in the following file:
  %1", cnn_multiclass.model.file_path)
    
    put_log("Saving the CNN MCC Model History...")
    saveRDS(cnn_multiclass.train_history,
            file = cnn_multiclass.train_history.file_path)
    
    put_log("The CNN MCC Model History has been trained 
and saved in the following file:
  %1", cnn_multiclass.train_history.file_path)
  }
}


put_log("The CNN MCC Model has been created and trained:

%1", capture.output(cnn_multiclass.model))


if(exists("cnn_multiclass.train_history")){
  plot(cnn_multiclass.train_history)
} 

put_end_date(start)

### Close Log ------------------------------------------------------------------
log_close()


