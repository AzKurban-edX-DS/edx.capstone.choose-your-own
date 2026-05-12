#%%%%%%%%%%%%%%%%%%%%%%%%%
# CNN-Based Binary Models
#%%%%%%%%%%%%%%%%%%%%%%%%%

### Convolutional Neural Network (CNN) -----------------------------------------
# Reference:
# Deep Learning Using R with keras (CNN)
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

#### Open log: Split (90% for Train) Dataset (x3d) -----------------------------
open_logfile(".ds.prepare.train&test.balanced_sets")
## Load Train+Test Data --------------------------------------------------------
if(!exists("x_cnn")) {
  stopifnot(file.exists(train.img28x28mx.array.file_path))
  x_cnn <- readRDS(train.img28x28mx.array.file_path)
  put_log("The Train Test Data has been loaded from the following backup file:
%1", train.img28x28mx.array.file_path)
}

put_log("The Train Test Data has the following structure:
%1", capture.output(str(x_cnn)))

#### Split Dataset -------------------------------------------------------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

sample_seed <- length(y.labels) # 39

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

set.seed(sample_seed)
split3d.list <- sample_train_test_sets.x3d(x_cnn)
str(split3d.list)

x_train <- split3d.list$train_set
dim(x_train)
#> [1] 16653    28    28
nrow(x_train)
#> [1] 16653

y_cnn.train <- as.factor(rownames(x_train))

y_cnn.train.cat <- to_categorical(y_cnn.train)
colnames(y_cnn.train.cat) <- y.labels


# Add channel into the dimension
x_cnn.train <- array_reshape(x_train, 
                             c(nrow(x_train), 
                               img_rows, 
                               img_cols, 
                               1))
x_test <- split3d.list$test_set
dim(x_test)
#> [1] 817379     28     28
nrow(x_test)
#> [1] 817379

y_cnn.test <- as.factor(rownames(x_test))

y_cnn.test.cat <- to_categorical(y_cnn.test)
colnames(y_cnn.test.cat) <- y.labels

# Add channel into the dimension
x_cnn.test <- array_reshape(x_test, 
                            c(nrow(x_test), 
                              img_rows, 
                              img_cols, 
                              1))
rm(split3d.list)
rm(x_train)
rm(x_test)

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x_cnn.train)
#> [1] 16653    28    28     1

# str(y_cnn.train)
length(y_cnn.train)
#> [1] 16653

dim(y_cnn.train.cat)
#> [1] 16653    39
str(y_cnn.train.cat)
head(y_cnn.train.cat[,1:30])

dim(x_cnn.test)
#> [1] 817379     28     28      1

# str(y_cnn.test)
length(y_cnn.test)
#> [1] [1] 817379

dim(y_cnn.test.cat)
#> [1] 817379     39
str(y_cnn.test.cat)
head(y_cnn.test.cat[,1:30])

### Close Log ------------------------------------------------------------------
log_close()

#### Open log: Build CNN Model -------------------------------------------------
open_logfile(".build-cnn-model")
#### Model building ------------------------------------------------------------
##### Define a CNN-Based Multiclass Classification model structure -------------------------------------
# Define a few parameters to be used in the CNN model
n.output <- 39
batch_size <- 128
num_classes <- 39
epochs <- 100
vld_split <- 0.2

#> Now we define a CNN model with two 2D convolutional layers with max pooling, 
#> and the 2nd layer with additonal dropout to prevent overfitting. 
#> Then flatten the output and use two dense layers to connect to the categoires 
#> of the image. [*]

cnn.multiclass.model <- keras_model_sequential(shape(28L, 28L, 1L)) |>
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

summary(cnn.multiclass.model)
# plot(cnn.multiclass.model)

# Similar to DNN model, we need to compile the defined CNN model. [*]

# Compile model
cnn.multiclass.model |> compile(
  loss = loss_categorical_crossentropy,
  # optimizer = optimizer_adadelta(),
  optimizer = keras3::optimizer_adamax(0.001),
  metrics = c('accuracy')
)

summary(cnn.multiclass.model)

#### Training CNN-Based Muliclass Classifier Model -----------------------------

#> Now, we can train the model with our processed data. 
#> Each epochs's history can be saved to track the progress. 
#> Please note, as we are not using GPU, it takes a few minutes to finish. 
#> Please be patient while waiting for the results. 
#> The training time can be significantly reduced if running on GPU. [*]
dim(x_cnn.train)
dim(y_cnn.train.cat)

dim(x_cnn.test)

str(y_cnn.test)
length(y_cnn.test)
dim(y_cnn.test.cat)
#> [1] 684467     39

put_log("Training the CNN-based Multiclass Classifier Model...")
start <- put_start_date()

str(x_cnn.train)
# Train model
cnn.multiclass.train_history <- cnn.multiclass.model |> 
  fit(x_cnn.train, 
      y_cnn.train.cat,
      epochs = epochs,
      batch_size = batch_size,
      validation_split = vld_split
  )
# acc: 0.8741
plot(cnn.multiclass.train_history)


put_log("Saving the CNN-based Multiclass Classifier Model...")
saveRDS(cnn.multiclass.train_history,
        file = cnn.multiclass.train_history.file_path)
put_log("Saving the CNN-based Multiclass Classifier Model...")

put_log("Saving pre-trained model...")
save_model(cnn.multiclass.model,
           filepath = cnn.multiclass.model.file_path,
           overwrite = TRUE)

put_log("The CNN Model has been trained and saved in the following file:
%1", cnn.multiclass.model.file_path)
put_end_date(start)

### Close Log ------------------------------------------------------------------
log_close()

#### Open log: Evaluate CNN Model -------------------------------------------------
open_logfile(".evaluate-cnn-model")
#### Evaluating CNN Model ----------------------------------------------

put_log("Evaluating CNN Model...")
start <- put_start_date()
cnn.eval.result <- cnn.multiclass.model |> evaluate(x_cnn.test, y_cnn.test.cat)
put_log("CNN Model evaluation result:
%1", capture.output(str(cnn.eval.result)))
# List of 2
#  $ accuracy: num 0.861
#  $ loss    : num 2.83

put_end_date(start)

# model prediction
put_log("CNN Model: constructing predictions...")

cnn_preds <- cnn.multiclass.model |> predict(x_cnn.test) 
put_log("CNN Model: predictions have been constructed.")
put_end_date(start)
# Time difference of 1.502232 mins

dim(cnn_preds)

colnames(cnn_preds) <- y.labels
head(cnn_preds[,1:5])

cnn_preds.ts <- as_tensor(cnn_preds)
str(cnn_preds.ts)
#> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>

cnn.predictionns <- cnn_preds.ts |> op_argmax(2)
str(cnn.predictionns)
cnn.predictionns
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(cnn.predictionns)
#> [1] 684467
cnn.prediction.values.idx <- cnn.predictionns$numpy()
head(cnn.prediction.values.idx)
cnn.prediction.values <- y.labels[cnn.prediction.values.idx]
head(cnn.prediction.values)

cnn.model.accuracy <- mean(cnn.prediction.values == y_cnn.test)
put_log("CNN Model accuracy: %1", cnn.model.accuracy)
# CNN Model accuracy: 0.920942675925091
#> For final test (expected value):
#> CNN Model accuracy: 0.908640379228615

err.idx <- which(cnn.prediction.values != as.integer(y_cnn.test))
length(err.idx)
# 94951
err.head.idx <- head(err.idx)



err.pred.values <- cnn.prediction.values[err.idx]
head(err.pred.values)

err.test.values <- y_cnn.test[err.idx]
head(err.test.values)

err.head.img <- x_cnn.test[err.head.idx,,,1]
dim(err.head.img)

par(mfrow = c(6, 1))
for(i in err.head.idx) {
   char.image(x_cnn.test[i,,,1])
}
par(mfrow = c(1,1))

#> [*] Reference: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html
### Close Log ------------------------------------------------------------------
log_close()

