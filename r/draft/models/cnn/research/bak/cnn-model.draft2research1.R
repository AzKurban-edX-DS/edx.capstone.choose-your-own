str(img28x28bin.list$img.list$T$img.list)

t.img.list <- img28x28bin.list$img.list$T$img.list
str(t.img.list)

t.img.flat_ls <- t.img.list |> img28x28.list2matrix("T")
str(t.img.flat_ls)

image(t.img.flat_ls[1:400,])

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

t.cnn_model <- keras_model_sequential(shape(28L, 28L, 1L)) |>
  layer_conv_2d(filters = 8L,
                kernel_size = c(2.2), 
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
  layer_dense(units = 2, activation = "sigmoid")

summary(t.cnn_model)
# plot(t.cnn_model)

# Similar to DNN model, we need to compile the defined CNN model. [*]

# Compile model
t.cnn_model |> compile(
  loss = 'binary_crossentropy',
  # optimizer = optimizer_adadelta(),
  # optimizer = keras3::optimizer_adamax(0.001),
  optimizer = optimizer_rmsprop(learning_rate = 0.0001),
  metrics = c('accuracy')
)
summary(t.cnn_model)

#### Training CNN Model --------------------------------------------------------

#> Now, we can train the model with our processed data. 
#> Each epochs's history can be saved to track the progress. 
#> Please note, as we are not using GPU, it takes a few minutes to finish. 
#> Please be patient while waiting for the results. 
#> The training time can be significantly reduced if running on GPU. [*]

str(x_cnn.1bl.train)

is_t.class <- as.character(y_cnn.1bl.train) == "T"
x_cnn.train <- x_cnn.1bl.train # [is_t.class,,,]
# x_cnn.train <- x_cnn.9bl.train
# x_cnn.train <- array_reshape(x_cnn.train, 
#                              c(nrow(x_cnn.train), 
#                                n.img_rows, 
#                                n.img_cols, 
#                                1))
dim(x_cnn.train)
str(x_cnn.train)

# y_cnn.train <- as.factor(as.character(y_cnn.1bl.train[is_t.class]))

y.ch <- as.character(y_cnn.1bl.train) 
str(y.ch)

y.T <- as.factor(ifelse(y.ch == "T", "T", "not_T"))
str(y.T)
length(y.T)

sum(as.character(y.T) == "T")


y_cnn.train <- y.T
# y_cnn.train <- y_cnn.1bl.train
# y_cnn.train <- y_cnn.9bl.train
str(y_cnn.train)
length(y_cnn.train)

# head(as.character(y_cnn.train))
# sum(as.character(y_cnn.train) == "T")
# T.y_cnn.train <- y_cnn.train[as.character(y_cnn.train) == "T"]
# str(T.y_cnn.train)
# length(T.y_cnn.train)


# str(y_cnn.1bl.train.cat)
# cat.is_t.class <- y_cnn.1bl.train.cat[,"T"] == 1
# str(cat.is_t.class)
# sum(cat.is_t.class)

y_cnn.train.cat <- to_categorical(y_cnn.train)
colnames(y_cnn.train.cat) <- c("not_T", "T")

# y_cnn.train.cat <- y_cnn.9bl.train.cat
dim(y_cnn.train.cat)
head(y_cnn.train.cat)
length(y_cnn.train.cat)


x_cnn.test <- x_cnn.9.test
# x_cnn.test <- x_cnn.1.test
dim(x_cnn.test)



y_cnn.test <-  
  as.factor(ifelse(as.character(y_cnn.9.test) == "T", "T", "not_T"))

# y_cnn.test <- y_cnn.9.test
# y_cnn.test <- y_cnn.1.test
str(y_cnn.test)
length(y_cnn.test)
sum(as.character(y_cnn.test) == "T")


y_cnn.test.cat <- to_categorical(y_cnn.test)
colnames(y_cnn.test.cat) <- c("not_T", "T")

# y_cnn.test.cat <- y_cnn.1.test.cat
dim(y_cnn.test.cat)
#> [1] 684467     39
head(y_cnn.test.cat)

put_log("Training the CNN Model...")
start <- put_start_date()

str(x_cnn.train)
# Train model
cnn.1bl.train_history <- t.cnn_model |> 
  fit(x_cnn.train, 
      y_cnn.train.cat,
      epochs = epochs,
      batch_size = 50,
      # validation_steps = 22 # 427/20
      validation_split = vld_split
  )
# acc: 0.8741

put_log("The CNN Model has been trained on `x_cnn.1bl.train` dataset.")
put_end_date(start)

plot(cnn.1bl.train_history)
#### Evaluating CNN Model ----------------------------------------------

put_log("Evaluating CNN Model...")
start <- put_start_date()
cnn.eval.result <- t.cnn_model |> evaluate(x_cnn.test, y_cnn.test.cat)
put_log("CNN Model evaluation result:
%1", capture.output(str(cnn.eval.result)))
# CNN Model evaluation result:
#   List of 2
# $ accuracy: num 0.963
# $ loss    : num 0.752
put_end_date(start)

# model prediction
put_log("CNN Model: constructing predictions...")
start <- put_start_date()
cnn_preds <- t.cnn_model |> predict(x_cnn.test) 
put_log("CNN Model: predictions have been constructed.")
put_end_date(start)
# Time difference of 1.502232 mins

dim(cnn_preds)

# colnames(cnn_preds) <- y.labels
colnames(cnn_preds) <- c("not_T", "T")
head(cnn_preds)
#head(cnn_preds[,1:5])

cnn_preds.ts <- as_tensor(cnn_preds)
str(cnn_preds.ts)
#> <tf.Tensor: shape=(817379, 39), dtype=float64, numpy=…>

cnn.predictionns <- cnn_preds.ts |> op_argmax(2)
str(cnn.predictionns)
cnn.predictionns
#> <tf.Tensor: shape=(817379), dtype=int32, numpy=array([1, 1, 1, ..., 1, 1, 1], shape=(817379,), dtype=int32)>
dim(cnn.predictionns)
#> [1] 817379
cnn.prediction.values.idx <- cnn.predictionns$numpy()
head(cnn.prediction.values.idx)
str(y.T)
length(y.T)
y.T.labels <- as.factor(c("not_T", "T"))
y.T.labels

cnn.prediction.values <- y.T.labels[cnn.prediction.values.idx]
head(cnn.prediction.values)
str(y_cnn.test)
head(y_cnn.test)

cnn.model.accuracy <- mean(cnn.prediction.values == y_cnn.test)
put_log("CNN Model accuracy: %1", cnn.model.accuracy)
# CNN Model accuracy: 0.962776141789794

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


