#%%%%%%%%%%%%%%%%%%%%%%%%%%
# Basic Deep Learning Model
#%%%%%%%%%%%%%%%%%%%%%%%%%%

### Open log: Split Train Dataset (x) -------------
open_logfile(".split.80%train.balanced_subset")
#### Split Train Dataset  (90% for Train set) ----------------------------------
# char_files.max4e3 <- 4e3 
# char_files.max4e3

dim.x <- dim(x)
dim.x
dim.x[1]
dim.x[2]

sample_seed <- dim.x[1]
sample_seed
shuffle_seed <- as.integer(sample_seed*test_ratio)
shuffle_seed

cache_file.path <- file.path(ds.subsets.path, "x.9bl.train(balanced).RData")
cache_file.path

start <- put_start_date()
cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

if (file.exists(cache_file.path)) {
  put_log1("Loading Split Train Data from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("Train Data list has been loaded from cache.")
} else {
  split.list <- sample_train_test_sets.mx(x, 
                                          sample_seed,
                                          test.ratio = 0.1,
                                          shuffle.seed = shuffle_seed)
  str(split.list)
  
  x.9bl.train <- split.list$train_set
  y.9bl.train <- as.factor(rownames(x.9bl.train))
  
  x.1.test <- split.list$test_set
  y.1.test <- as.factor(rownames(x.1.test))
  
  
  #   x0.1.test.list <- splitDataset(split.list$test_set, 9)
  #   put_log1("Test dataset list structure:
  # %1", capture.output(str(x0.1.test.list)))
  
  put_log1("Caching data in the file
%1 ...", cache_file.path)
  
  save(x.9bl.train,
       y.9bl.train,
       x.1.test,
       y.1.test,
       file = cache_file.path)
  
  put_log1("The Train Data Subset objects has been cached in file:
`%1`", cache_file.path)
  
  rm(split.list)
}

stopCluster(cl)
stopImplicitCluster()
put_end_date(start)

dim(x.9bl.train)
#> [1] 16653   784
str(x.9bl.train)

str(y.9bl.train)
length(y.9bl.train)

str(x.1.test)
str(y.1.test)
length(y.1.test)
#> [1] 817379


### Close Log ------------------------------------------------------------------
log_close()

### Basic DL Classifier -----------------------------------------------------------
# Reference:
# MNIST Handwritten Digit Recognition in Keras
# https://nextjournal.com/gkoehler/digit-recognition-with-keras

#### Converting labels factor to categorical -----------------------------------
# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.
# https://www.r-bloggers.com/2021/02/deep-learning-with-r-and-keras-build-a-handwritten-digit-classifier-in-10-minutes/

y.9bl.train.cat <- to_categorical(y.9bl.train)
colnames(y.9bl.train.cat) <- y.labels
dim(y.9bl.train.cat)
str(y.9bl.train.cat)
head(y.9bl.train.cat)
# max(y.9bl.train.cat)

y.1.test.cat <- to_categorical(y.1.test)
colnames(y.1.test.cat) <- y.labels
dim(y.1.test.cat)
str(y.1.test.cat)
head(y.1.test.cat)

#### Open log: Building Basic DL Model -----------------------------------------
open_logfile("dl.basic-model")
#### DL Model building on dataset: `dl.model`: `x0.1.dl.model` ---------

if(!dir.exists(dl.keras3.path))
  dir.create(dl.keras3.path)

cache_file.path <- file.path(dl.keras3.path, 
                             "dl.x.9bl.train.model.RData")

if (file.exists(cache_file.path)) {
  put_log("Loading `DL Keras3` model from cache file: 
%1", cache_file.path)
  
  load(cache_file.path)
  put_log("`DL Keras3` model has been loaded from cache.")
  
} else {
  n.input_shape <- ncol(x.9bl.train)
  # 784
  
  n.output <- length(y.labels)
  # 39
  
  n.hl.units <- ceiling(n.input_shape*2/3+n.output)
  # 562
  
  dl.model <- keras_model_sequential() |>
    layer_dense(units = n.hl.units, activation = "relu", 
                input_shape = c(n.input_shape)) |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.hl.units, activation = "relu") |>
    layer_dropout(rate = 0.25) |> 
    layer_dense(units = n.output, activation = "softmax")
  
  summary(dl.model)
  
  dl.model |> compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(),
    metrics = c("accuracy")
  )
  
  put_log("Training the Basic DL Model...")
  start <- put_start_date()
  
  dl.x.9bl.train.history <- dl.model |> 
    fit(x.9bl.train, 
        y.9bl.train.cat, 
        epochs = 100, 
        batch_size = 512, 
        validation_split = 0.15)
  
  put_log("The Basic DL Model has been trained on `x.9bl.train` dataset.")
  put_end_date(start)
  
  put_log("Saving `DL Keras3` model to the cache file...")
  start <- put_start_date()
  
  save(dl.model,
       dl.x.9bl.train.history,
       file = cache_file.path)
  
  put_log("The `DL Keras3` model has been saved to the cache file: 
%1", cache_file.path)
  put_log("Training Basic DL Model task has been completed on the Training Set 
(90% balanced sample of the dataset).")
  put_end_date(start)
  # Time difference of 38.48235 mins
}

plot(dl.x.9bl.train.history)
str(dl.x.9bl.train.history)
#### DL Basic Model Evaluation ----------------------------------------------
put_log("Evaluating DL Model...")
start <- put_start_date()
dl.eval.result <- dl.model |> evaluate(x.1.test, y.1.test.cat)
put_log("DL Model evaluation result:
%1", capture.output(str(dl.eval.result)))
# List of 2
#  $ accuracy: num 0.796
#  $ loss    : num 1.52

put_end_date(start)
# Time difference of 1.668308 mins

start <- put_start_date()
dl.preds <- dl.model |> predict(x.1.test) 
put_end_date(start)
# Time difference of  mins

colnames(dl.preds) <- y.labels
head(dl.preds[,1:5])
#                 #            $            &            @            0
# [1,] 8.469580e-25 2.824278e-25 1.338040e-31 1.180656e-36 3.503761e-16
# [2,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
# [3,] 9.542136e-15 3.782388e-15 1.619873e-16 3.058267e-19 4.522308e-08
# [4,] 7.770129e-13 1.467184e-19 8.129414e-25 4.881509e-21 9.999547e-01
# [5,] 4.731567e-38 0.000000e+00 0.000000e+00 0.000000e+00 5.828601e-27
# [6,] 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00

dim(dl.preds)
#> [1] 684467     39

dl.preds.ts <- as_tensor(dl.preds)
str(dl.preds.ts)
#> <tf.Tensor: shape=(684467, 39), dtype=float64, numpy=…>

dl.predictions <- dl.preds.ts |> op_argmax(2)
dl.predictions
#> tf.Tensor([13  4 21 ... 19  5  1], shape=(684467), dtype=int32)
dim(dl.predictions)
#> [1] 684467
# dl.predictions$numpy()


# y.1.test
# as.integer(y.1.test)

dl.model.accuracy <- mean(dl.predictions$numpy() == as.integer(y.1.test))
put_log("DL Model accuracy: %1",dl.model.accuracy)

# [1] 0.7955373

##### Close Log ------------------------------------------------------------------
log_close()

# ---------------------------
# Reference:
#
# 
start <- put_start_date()
put_end_date(start)






























