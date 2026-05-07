### Convolutional Neural Network (CNN) -----------------------------------------
# Reference:
# Deep Learning Using R with keras (CNN)
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html

#### Open log: Prepare CNN Datasets -------------
open_logfile(".prepare-cnn-datasets")
#### Prepare CNN Datasets ------------------------------------------------------
# train.img28x28bin.list.file_path <- file.path(train.data.path, "train.img28x28bin.list.RData")
# train.img28x28bin.list.file_path


cnn.train.data.path <- file.path(dl.keras3.path, "cnn")

if(!dir.exists(cnn.train.data.path))
  dir.create(cnn.train.data.path)

cnn.train.data.cache.file_path <- file.path(cnn.train.data.path, "xy_cnn.data_cache.RData")
cnn.train.data.cache.file_path

if(!file.exists(cnn.train.data.cache.file_path)){
  if(!exists("img28x28bin.list")) {
    if(!file.exists(train.img28x28bin.list.file_path)) {
      stop("Cache file for Binary Image 28x28 Matrix list DOES NOT EXIST!")
    }
    start <- put_start_date()
    put_log("Loading Binary Image 28x28 Matrix list from the backup file...")
    load(train.img28x28bin.list.file_path)
    put_log("The Binary Image 28x28 Matrix list has been loaded from the following file:
%1", train.img28x28bin.list.file_path)
    put_end_date(start)
  }
  put_log("The Binary Image 28x28 Matrix list object summary:
%1", capture.output(summary(img28x28bin.list)))
  
  y.labels <- img28x28bin.list$label.list
  y.labels
  
  img.nested_list <- img28x28bin.list$img.list
  length(img.nested_list)
  names(img.nested_list)
  
  start <- put_start_date()
  put_log("Combining nested list of images to list of arrays...")
  
  img28x28mx.list <- lapply(names(img.nested_list), function(label){
    item <- img.nested_list[[label]]
    img.array <- abind(item$img.list, rev.along = 3)
    dimnames(img.array) <- list(base::rep(label, 
                                          times = length(item$img.list)),
                                NULL,
                                NULL)
    img.array
  })
  
  put_log("Combined image data matrix has been created with the following structure:
  %1", capture.output(str(img28x28mx.list)))
  put_end_date(start)
  # Time difference of 2.757908 mins
  
  rm(img28x28bin.list)
  rm(img.nested_list)
  
  start <- put_start_date()
  put_log("Combining image list to array...")
  x_cnn <- abind(img28x28mx.list, along = 1)
  
  put_log("Combined image matrix array has been created with the following dimentions:
  %1", capture.output(dim(x_cnn)))
  # char.image(x_cnn[2,,])
  put_end_date(start)
  # Time difference of 18.09567 secs
  
  rm(img28x28mx.list)
  
  put_log("Caching data in the file
%1 ...", cnn.train.data.cache.file_path)
  start <- put_start_date()
  save(x_cnn,
       y_cnn,
       y.labels,
       file = cnn.train.data.cache.file_path)
  
  put_log("The Train Data Subset objects has been cached in file:
`%1`", cnn.train.data.cache.file_path)
  put_end_date(start)
  # Time difference of 1.813179 mins
  
  
} else {
  put_log("Loading CNN Train Data from cache file: 
%1", cnn.train.data.cache.file_path)
  
  load(cnn.train.data.cache.file_path)
  put_log("Train Data list has been loaded from cache.")
}

str(x_cnn)

dim.x_cnn <- dim(x_cnn)
dim.x_cnn
#> [1] 834032     28     28
nrow(x_cnn)
#> [1] 834032

# Input image dimensions
img_rows <- dim.x_cnn[2]
img_rows
img_cols <- dim.x_cnn[3]
img_cols

y_cnn <- as.factor(rownames(x_cnn))
str(y_cnn)
length(y_cnn)

first_G.idx <- which(y_cnn == "G")[1]
first_G.idx
#> [1] 598137

char.image(x_cnn[first_G.idx,,])
char.image(x_cnn[first_G.idx - 1,,])

# # Add channel into the dimension
# x3d <- array_reshape(x_cnn, c(dim.x_cnn[1], dim.x_cnn[2], dim.x_cnn[3], 1))
# dim(x3d)

# rownames(x3d) <- as.character(y)
# str(x3d)



input_shape <- c(dim.x_cnn[2], dim.x_cnn[3], 1)
input_shape
#> [1] 28 28  1
#### Close Log ------------------------------------------------------------------
log_close()

#### Open log: Split Train Dataset (x3d) -------------
open_logfile(".split3d.10%train.balanced_subset")
#### Split (x3d) (10% for Train) Dataset  (10% for Train set) -----------------------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

sample_seed <- 1
shuffle_seed <- 2

cache_file.path <- file.path(cnn.train.data.path, "cnn(.1train)-datasets.RData")
cache_file.path

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)
if (file.exists(cache_file.path)) {
  put_log("Loading Split Train Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Train Data list has been loaded from cache.")
} else {
  
  split3d.list <- sample_train_test_sets.x3d(x_cnn, 
                                             sample_seed,
                                             test.ratio = 0.9,
                                             shuffle.seed = shuffle_seed)
  str(split3d.list)
  
  x_train <- split3d.list$train_set
  dim(x_train)
  #> [1] 16653    28    28
  nrow(x_train)
  #> [1] 16653
  
  y_cnn.1bl.train <- as.factor(rownames(x_train))
  str(y_cnn.1bl.train)
  length(y_cnn.1bl.train)
  
  y_cnn.1bl.train.cat <- to_categorical(y_cnn.1bl.train)
  colnames(y_cnn.1bl.train.cat) <- y.labels
  
  
  # Add channel into the dimension
  x_cnn.1bl.train <- array_reshape(x_train, 
                                   c(nrow(x_train), 
                                     img_rows, 
                                     img_cols, 
                                     1))
  rownames(x_cnn.1bl.train) <- rownames(x_train)
  str(x_cnn.1bl.train)
  x_test <- split3d.list$test_set
  dim(x_test)
  #> [1] 817379     28     28
  nrow(x_test)
  #> [1] 817379
  
  y_cnn.9.test <- as.factor(rownames(x_test))
  str(y_cnn.9.test)
  
  y_cnn.9.test.cat <- to_categorical(y_cnn.9.test)
  colnames(y_cnn.9.test.cat) <- y.labels
  str(y_cnn.9.test.cat)
  
  # Add channel into the dimension
  x_cnn.9.test <- array_reshape(x_test, 
                                c(nrow(x_test), 
                                  img_rows, 
                                  img_cols, 
                                  1))
  rownames(x_cnn.9.test) <- rownames(x_test)
  str(x_cnn.9.test)
  
  start <- put_start_date()
  put_log("Caching data in the file
%1 ...", cache_file.path)
  
  save(x_cnn.1bl.train,
       y_cnn.1bl.train,
       y_cnn.1bl.train.cat,
       x_cnn.9.test,
       y_cnn.9.test,
       y_cnn.9.test.cat,
       file = cache_file.path)
  
  put_log("The Train Data Subset objects has been cached in file:
`%1`", cache_file.path)
  put_end_date(start)
 # Time difference of 48.37844 secs
  
  rm(split3d.list)
  rm(x_train)
  rm(x_test)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x_cnn.1bl.train)
#> [1] 16653    28    28     1

# str(y_cnn.1bl.train)
length(y_cnn.1bl.train)
#> [1] 16653

dim(y_cnn.1bl.train.cat)
#> [1] 16653    39
str(y_cnn.1bl.train.cat)
head(y_cnn.1bl.train.cat[,1:30])

dim(x_cnn.9.test)
#> [1] 817379     28     28      1

# str(y_cnn.9.test)
length(y_cnn.9.test)
#> [1] [1] 817379

dim(y_cnn.9.test.cat)
#> [1] 817379     39
str(y_cnn.9.test.cat)
head(y_cnn.9.test.cat[,1:30])

### Close Log ------------------------------------------------------------------
log_close()

#### Open log: Split (90% for Train) Dataset (x3d) -------------
open_logfile(".split3d.90%train.balanced_subset")
#### Split (x3d) Train Dataset  (90% for Train set) -----------------------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

sample_seed <- 9
shuffle_seed <- 1

cache_file.path <- file.path(ds.subsets.path, "cnn(.9train)-datasets.RData")
cache_file.path

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)
if (file.exists(cache_file.path)) {
  put_log("Loading Split Train Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Train Data list has been loaded from cache.")
} else {
  
  split3d.list <- sample_train_test_sets.x3d(x_cnn, 
                                             sample_seed,
                                             test.ratio = 0.1,
                                             shuffle.seed = shuffle_seed)
  str(split3d.list)
  
  x_train <- split3d.list$train_set
  dim(x_train)
  #> [1] 16653    28    28
  nrow(x_train)
  #> [1] 16653
  
  y_cnn.9bl.train <- as.factor(rownames(x_train))
  
  y_cnn.9bl.train.cat <- to_categorical(y_cnn.9bl.train)
  colnames(y_cnn.9bl.train.cat) <- y.labels
  
  
  # Add channel into the dimension
  x_cnn.9bl.train <- array_reshape(x_train, 
                                   c(nrow(x_train), 
                                     img_rows, 
                                     img_cols, 
                                     1))
  x_test <- split3d.list$test_set
  dim(x_test)
  #> [1] 817379     28     28
  nrow(x_test)
  #> [1] 817379
  
  y_cnn.1.test <- as.factor(rownames(x_test))
  
  y_cnn.1.test.cat <- to_categorical(y_cnn.1.test)
  colnames(y_cnn.1.test.cat) <- y.labels
  
  # Add channel into the dimension
  x_cnn.1.test <- array_reshape(x_test, 
                                c(nrow(x_test), 
                                  img_rows, 
                                  img_cols, 
                                  1))
  start <- put_start_date()
  put_log("Caching data in the file
%1 ...", cache_file.path)
  
  save(x_cnn.9bl.train,
       y_cnn.9bl.train,
       y_cnn.9bl.train.cat,
       x_cnn.1.test,
       y_cnn.1.test,
       y_cnn.1.test.cat,
       file = cache_file.path)
  
  put_log("The Train Data Subset objects has been cached in file:
`%1`", cache_file.path)
  put_end_date(start)
  
  rm(split3d.list)
  rm(x_train)
  rm(x_test)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x_cnn.9bl.train)
#> [1] 16653    28    28     1

# str(y_cnn.9bl.train)
length(y_cnn.9bl.train)
#> [1] 16653

dim(y_cnn.9bl.train.cat)
#> [1] 16653    39
str(y_cnn.9bl.train.cat)
head(y_cnn.9bl.train.cat[,1:30])

dim(x_cnn.1.test)
#> [1] 817379     28     28      1

# str(y_cnn.1.test)
length(y_cnn.1.test)
#> [1] [1] 817379

dim(y_cnn.1.test.cat)
#> [1] 817379     39
str(y_cnn.1.test.cat)
head(y_cnn.1.test.cat[,1:30])

### Close Log ------------------------------------------------------------------
log_close()

#### Open log: Build CNN Model -------------------------------------------------
open_logfile(".build-cnn-model")
#### Model building ------------------------------------------------------------
##### Define a CNN model structure -------------------------------------
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

cnn_model <- keras_model_sequential(shape(28L, 28L, 1L)) |>
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

summary(cnn_model)
# plot(cnn_model)

# Similar to DNN model, we need to compile the defined CNN model. [*]

# Compile model
cnn_model |> compile(
  loss = loss_categorical_crossentropy,
  # optimizer = optimizer_adadelta(),
  optimizer = keras3::optimizer_adamax(0.001),
  metrics = c('accuracy')
)
summary(cnn_model)

#### Training CNN Model --------------------------------------------------------

#> Now, we can train the model with our processed data. 
#> Each epochs's history can be saved to track the progress. 
#> Please note, as we are not using GPU, it takes a few minutes to finish. 
#> Please be patient while waiting for the results. 
#> The training time can be significantly reduced if running on GPU. [*]

# x_cnn.train <- x_cnn.1bl.train
x_cnn.train <- x_cnn.9bl.train
dim(x_cnn.train)

# y_cnn.train.cat <- y_cnn.1bl.train.cat
y_cnn.train.cat <- y_cnn.9bl.train.cat
dim(y_cnn.train.cat)

# x_cnn.test <- x_cnn.9.test
x_cnn.test <- x_cnn.1.test
dim(x_cnn.test)

# y_cnn.test <- y_cnn.9.test
y_cnn.test <- y_cnn.1.test
str(y_cnn.test)
length(y_cnn.test)

# y_cnn.test.cat <- y_cnn.9.test.cat
y_cnn.test.cat <- y_cnn.1.test.cat
dim(y_cnn.test.cat)
#> [1] 684467     39

put_log("Training the CNN Model...")
start <- put_start_date()

str(x_cnn.train)
# Train model
cnn.1bl.train_history <- cnn_model |> 
  fit(x_cnn.train, 
      y_cnn.train.cat,
      epochs = epochs,
      batch_size = batch_size,
      validation_split = vld_split
  )
# acc: 0.8741

put_log("The CNN Model has been trained on `x_cnn.1bl.train` dataset.")
put_end_date(start)

#### Evaluating CNN Model ----------------------------------------------

put_log("Evaluating CNN Model...")
start <- put_start_date()
cnn.eval.result <- cnn_model |> evaluate(x_cnn.test, y_cnn.test.cat)
put_log("CNN Model evaluation result:
%1", capture.output(str(cnn.eval.result)))
# List of 2
#  $ accuracy: num 0.861
#  $ loss    : num 2.83

put_end_date(start)

# model prediction
put_log("CNN Model: constructing predictions...")
start <- put_start_date()
cnn_preds <- cnn_model |> predict(x_cnn.test) 
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
# DL Model accuracy: 0.861277461148602

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

