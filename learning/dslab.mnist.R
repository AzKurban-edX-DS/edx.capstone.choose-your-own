# 21 Working with Matrices in R ------------------------------------------------
# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#sec-mnist

## Setup -----------------------------------------------------------------------

if(!require("logr")) 
  install.packages("logr")

if(!require(matrixStats))
  install.packages("matrixStats")

if(!require(dslabs))
  install.packages("dslabs")
if(!require(tidyverse))
  install.packages("tidyverse")
if(!require(prodlim))
  install.packages("prodlim")
if(!require(caret))
  install.packages("caret")

if(!require(randomForest))
  install.packages("randomForest")

# if(!require())
#   install.packages("")

if(!require(doParallel))
  install.packages("doParallel")

library(matrixStats)
library(dslabs)
library(tidyverse)

library(prodlim)
library(caret)
library(randomForest)

library(logr)
library(doParallel)

N_pcCores <- detectCores() - 1   # it is convention to leave 1 core for the OS
N_pcCores

r.path <- "r"

# support_scripts.folder <- "support-scripts"
support_functions.folder <- "support-functions"

# support_scripts.path <- file.path(r.src.path, support_scripts.folder)
support_functions.path <- file.path(r.path, support_functions.folder)


### Define Logging Functions ---------------------------------------------------
log_func_script.file_path <- file.path(support_functions.path, "logging-functions.R")

source(log_func_script.file_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)


## 21.2 Case Study: MNIST ------------------------------------------
library(dslabs)

mnist <- dslabs::read_mnist()
str(mnist)

x <- mnist$train$images
y <- mnist$train$labels
str(y)


## 21.3 Matrix Operations in R --------------------------------------------------

### Dimentions of a matrix --------------------------

dim(x)
nrow(x)
ncol(x)

### Subsetting --------------------------------------

x[300, 100]
x[1:300, 1:100]

x[1:10,]


#> If we subset just one row or just one column, the resulting object is no longer a matrix. 
#> For example notice what happens here: 
x[300,]
dim(x[300,])


# To avoid this, we can use the drop argument:
dim(x[100,,drop = FALSE])
#> [1]   1 784

### The transpose ----------------------------------

dim(x)
#> [1] 60000   784
dim(t(x))
#> [1]   784 60000

### Row and column summaries ----------------------------

avgs <- apply(x, 1, mean)
str(avgs)

sds <- apply(x, 1, sd)
str(sds)



row_avg <- rowMeans(x)
str(row_avg)

library(matrixStats)
row_sds <- rowSds(x)

### Conditional filtering l----------------------------


matrix(1:15, 3, 5)[,c(FALSE, TRUE, TRUE, FALSE, TRUE)]
#>      [,1] [,2] [,3]
#> [1,]    4    7   13
#> [2,]    5    8   14
#> [3,]    6    9   15

### Indexing with matrices

mat <- matrix(1:15, 3, 5)
mat[mat > 6 & mat < 12] <- 0
mat
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    4    0    0   13
#> [2,]    2    5    0    0   14
#> [3,]    3    6    0   12   15


## 21.4 Vectorization for matrices ---------------------------------------------

### Matrix–vector operations -----------------------------

#> A common example of matrix vectorization is centering rows: subtracting the 
#> mean of each row from its corresponding entries. Although this can be done 
#> with the `scale()` function

scale(x, scale = FALSE)

# we can center each row of x with this single line of code:

x - rowMeans(x)

### The `sweep` function ------------------------------

#> If we instead want to scale the columns, the following code will not work as expected 
#> because subtraction is applied row-wise:
  
x - colMeans(x)

#> One workaround is to transpose the matrix, apply the operation, and transpose back:
  
  t(t(x) - colMeans(x))
  
# Using `sweep` to subtract each column’s mean we write:
  
  sweep(x, MARGIN = 2, STATS = colMeans(x), FUN = "-")  
  

# Visualization ------------------------------------------

grid <- matrix(x[3,], 28, 28)
grid

image(grid)
image(grid[, 28:1])
image(1:28, 1:28, grid[, 28:1])

# Do some digits require more ink to write than others? --------------------
avg <- rowMeans(x)
str(avg)

boxplot(avg ~ y)

# Are some pixels uninformative? ------------------------

sds <- colSds(x)
str(sds)
hist(sds, breaks = 30, main = "SDs")

image(1:28, 1:28, matrix(sds, 28, 28)[, 28:1])
# We see that there is little variation in the corners

#> So if we wanted to remove uninformative predictors from our matrix, 
#> we could write this one line of code:
new_x <- x[,colSds(x) > 60]
dim(new_x)
#> [1] 60000   322

# Can we remove smudges? -------------------------------------------------------

# We will first look at the distribution of all pixel values.
hist(as.vector(x), breaks = 30, main = "Pixel intensities")

#> This shows a clear dichotomy which is explained as parts of the image with ink and parts without. 
#> If we think that values below, say, 50 are smudges, we can quickly make them zero 
#> using what we learned in Section 21.3.7:
new_x <- x
new_x[new_x < 50] <- 0

# Binarize the data ------------------------------------------------------------

#> The histogram above seems to suggest that this data is mostly binary. 
#> A pixel either has ink or does not. Applying what we learned in Section 21.3.7, 
#> we can binarize the data using just matrix operations:

bin_x <- x
bin_x[bin_x < 255/2] <- 0 
bin_x[bin_x > 255/2] <- 1

#> We can also convert to a matrix of logicals and then coerce to numbers 
#> using what we learned in Section 21.4:


head(y)
bin_x[1,]

### Standardize the digits

#> Finally, we will scale each column to have the same average and standard deviation.
#> Using what we learned in Section 21.4 implies that we can scale each row of a matrix as follows:

(x - rowMeans(x))/rowSds(x)

#> And for the columns we can combine two calls to sweep:And for the columns 
#> we can combine two calls to sweep:

sweep(sweep(x, 2, colMeans(x)), 2, colSds(x), FUN = "/")

#> Note we can use this to standardize rows as well by replacing 2 with 1 and use rowMeans and rowSds.

# Finally, note that equivalent functionality can often be achieved with:
  
t(scale(t(x)))

# This version can be faster for large matrices, though it’s less explicit about what’s happening.

## 22.3 Distance ---------------------------------------------------------------

# Compute the distance matrix:

# d <- dist(x)
# class(d)

  
# 31 Building Machine Learning Model -------------------------------------------

## 31.1 Case study: handwritten digit recognition ------------------------------

### Sample of 1000 elements -------------------
set.seed(1990)
index <- sample(nrow(mnist$train$images), 10000)
x1e3 <- mnist$train$images[index,]
str(x1e3)

y1e3 <- factor(mnist$train$labels[index])
str(y1e3)

index <- sample(nrow(mnist$test$images), 1000)
x1e3_test <- mnist$test$images[index,]
y1e3_test <- factor(mnist$test$labels[index])

colnames(x1e3) <- 1:ncol(mnist$train$images)
colnames(x1e3_test) <- colnames(x1e3)


## 31.3 Preprocessing --------------------------------

# library(matrixStats)
hist(colSds(x), breaks = 256)

#> The caret packages includes a function that recommends features to be removed 
#> due to near zero variance:

nzv <- nearZeroVar(x)
str(nzv)

#> We can see the columns recommended for removal are the near the edges:

image(matrix(1:784 %in% nzv, 28, 28))

#> Below is an example demonstrating how to remove predictors with near-zero variance 
#> and then center the remaining predictors:

str(x1e3)  
pp <- preProcess(x1e3, method = c("nzv", "center"))
str(pp)

centered_subsetted_x_test <- predict(pp, newdata = x_test)
dim(centered_subsetted_x_test)
#> [1] 1000  252

str(centered_subsetted_x_test)

## 31.5 k-nearest neighbors ----------------------------------------------------

### Optimizing `k` --------

train_knn <- train(x1e3, y1e3, method = "knn", 
                   preProcess = "nzv",
                   trControl = trainControl("cv", number = 20, p = 0.95),
                   tuneGrid = data.frame(k = seq(1, 7, 2)))
str(train_knn)

y_hat_knn <- predict(train_knn, x_test, type = "raw")
str(y_hat_knn)
length(y_hat_knn)

mean(y_hat_knn == y_test)
#> [1] 0.952

test_result1e3 <- list(predicted = y_hat_knn,
                              actual = y_test,
                              img = x_test)

str(test_result1e3)


show_digit <- function(idx) {
  
  print_log1("predicted: %1", y_hat_knn[idx])
  print_log1("actual: %1", y_test[idx])

  grid <- matrix(x_test[idx,], 28, 28)
  image(grid[, 28:1])
  # image(grid)
  #image(1:28, 1:28, grid[, 28:1])
}

show_digit(3)

## Train and test on entire `mnist` dataset ------------------------------------

x <- mnist$train$images
y <- mnist$train$labels

x_test <- mnist$test$images
y_test <- mnist$test$labels

colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(x)

### k-nearest neighbors ----------------------------------------------------
start_date <- print_start_date()
train_knn <- train(x, y, method = "knn", 
                   preProcess = "nzv",
                   trControl = trainControl("cv", number = 20, p = 0.95),
                   tuneGrid = data.frame(k = seq(1, 7, 2)))

y_hat_knn <- predict(train_knn, x_test, type = "raw")
print_end_date(start = start_date)
#> Time difference of 1.510466 hours

mean(y_hat_knn == y_test)
#> [1] 0.9275

### Dimension reduction with PCA

library(doParallel)

#### Start Do Parallel -----------------------
nc <- detectCores() - 1   # it is convention to leave 1 core for the OS
nc

cl <- makeCluster(nc)
registerDoParallel(cl)

#> As an example, suppose we want to retain the smallest number of PCs that explain 
#> at least 90% of the variability:

start <- print_start_date()
pca <- prcomp(x)
str(pca)

p <- which(cumsum(pca$sdev^2) / sum(pca$sdev^2) >= 0.9)[1]
p 
#> [1] 87

print_end_date(start)
#> Time difference of 1.399422 mins

str(y)
y <- factor(y)
str(y)

start <- print_start_date()
#> We can now re-run our algorithm using only these 87 transformed features:
fit_knn_pca <- knn3(pca$x[, 1:p], y, k = train_knn$bestTune)
print_end_date(start)
#> Time difference of 0.01428485 secs

#> A critical point when using PCA for prediction is that the PCA transformation 
#> must be learned only from the training set. If we use the validation or test set 
#> to compute principal components, or even to compute the means used for centering, 
#> we inadvertently leak information from those sets into the training process, 
#> leading to overtraining.

#> To avoid this, we compute the necessary centering and rotation matrices on the training set:

start <- print_start_date()
newdata <- sweep(x_test, 2, colMeans(x)) %*% pca$rotation[, 1:p]
y_hat_knn_pca <- predict(fit_knn_pca, newdata, type = "class")
print_end_date(start)
#> Time difference of 45.38669 secs

stopCluster(cl)
stopImplicitCluster()
#### End Do Parallel -----------------------

mean(y_hat_knn_pca == y_test)
#> [1] 0.9744


#> Here is how we modify our earlier code to let caret perform PCA during preprocessing:

#### Start Do Parallel -----------------------
nc <- detectCores() - 1   # it is convention to leave 1 core for the OS
nc

cl <- makeCluster(nc)
registerDoParallel(cl)

start <- print_start_date()
train_knn_pca <- train(x, y, method = "knn", 
                       preProcess = c("nzv", "pca"),
                       trControl = trainControl("cv", number = 20, p = 0.95,
                                                preProcOptions = list(thresh = 0.9)),
                       tuneGrid = data.frame(k = seq(1, 7, 2)))

y_hat_knn_pca <- predict(train_knn_pca, x_test, type = "raw")
mean(y_hat_knn_pca == y_test)
#> [1] 0.9741

print_end_date(start)
#> Time difference of 11.14748 mins

stopCluster(cl)
stopImplicitCluster()
#### End Do Parallel -----------------------

## 31.6 Random Forest ----------------------------------------------------------

#> With the random forest algorithm several parameters can be optimized, but the main one is `mtry`, 
#> the number of predictors that are randomly selected for each tree. 
#> This is also the only tuning parameter that the `caret` function `train` permits 
#> when using the default implementation from the randomForest package.

#### Start Do Parallel -----------------------
nc <- detectCores() - 1   # it is convention to leave 1 core for the OS
nc

cl <- makeCluster(N_pcCores)
registerDoParallel(cl)

start <- print_start_date()
library(randomForest)

train1e3_rf <- train(x1e3, y1e3, method = "rf", 
                  preProcess = "nzv",
                  tuneGrid = data.frame(mtry = seq(5, 15)))
#str(train1e3_rf)
stopCluster(cl)
stopImplicitCluster()
print_end_date(start)
#> Time difference of 11.14748 mins

#### End Do Parallel -----------------------

plot(train1e3_rf)

# Now that we have optimized our algorithm, we are ready to fit our final model:
# y1e3_hat_rf <- predict(train1e3_rf, x1e3_test, type = "raw")
y_hat_rf <- predict(train1e3_rf, x_test, type = "raw")

# As with `kNN`, we also achieve high accuracy:
#mean(y1e3_hat_rf == y1e3_test)
#> [1] 0.954

mean(y_hat_rf == y_test)
#> [1] 0.9518

#> By optimizing some of the other algorithm parameters, we can achieve even higher accuracy.

## Testing and improving computation time --------------------------------------

#> The default method for estimating accuracy used by the train function is 
#> to test prediction on 25 bootstrap samples. 
#> This can result in long compute times. 

nzv <- nearZeroVar(x)
str(nzv)

system.time({fit_rf <- randomForest(x[, -nzv], y,  mtry = 9)})
#>    user  system elapsed 
#> 1247.32    3.98 1254.93 

# use this to estimate the total time for the 250 iterations. In this case it will be several hours.


#### Start Do Parallel with Entire Dataset -----------------------
# nc <- detectCores() - 1   # it is convention to leave 1 core for the OS
# nc
# 
# cl <- makeCluster(nc)
# registerDoParallel(cl)
# 
# start <- print_start_date()
# library(randomForest)
# 
# train_rf <- train(x, y, method = "rf", 
#                   preProcess = "nzv",
#                   tuneGrid = data.frame(mtry = seq(5, 15)))
# 
# # Now that we have optimized our algorithm, we are ready to fit our final model:
# y_hat_rf <- predict(train_rf, x_test, type = "raw")
# 
# 
# print_end_date(start)
# #> Time difference of 11.14748 mins
# 
# stopCluster(cl)
# stopImplicitCluster()
#### End Do Parallel -----------------------

# As with `kNN`, we also achieve high accuracy:
# mean(y_hat_rf == y_test)
#> [1] 




