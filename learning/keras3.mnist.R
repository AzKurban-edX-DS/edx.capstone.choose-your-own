# Reference: 
#> Deep Learning with R and Keras: Build a Handwritten Digit Classifier in 10 Minutes
# https://www.appsilon.com/post/r-keras-mnist#:~:text=do%20that%20next.-,Model%20Training,function%20to%20train%20the%20model.

library(keras3)

c(c(x_train, y_train), c(x_test, y_test)) %<-% keras3::dataset_mnist()
head(x_train)

str(x_train)
# int [1:60000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
dim(x_train)
# 60000    28    28

X784_train <- array_reshape(x_train, c(nrow(x_train), 784))

dim(X784_train)
#> [1] 60000   784