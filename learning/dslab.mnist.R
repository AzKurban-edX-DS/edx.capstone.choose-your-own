# 21 Working with Matrices in R ------------------------------------------------
# Reference: https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#sec-mnist

## 21.2 Case Study: MNIST ------------------------------------------

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

library(matrixStats)
library(dslabs)
library(tidyverse)
library(prodlim)
library(caret)

mnist <- read_mnist()
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



# data(mnist_27)
# str(mnist_27)


image(grid)

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

set.seed(1990)
index <- sample(nrow(mnist$train$images), 10000)
x10e3 <- mnist$train$images[index,]
str(x10e3)

y10e3 <- factor(mnist$train$labels[index])
str(y10e3)

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])

colnames(x10e3) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(x10e3)


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

str(x10e3)  
pp <- preProcess(x10e3, method = c("nzv", "center"))
str(pp)

centered_subsetted_x_test <- predict(pp, newdata = x_test)
dim(centered_subsetted_x_test)
#> [1] 1000  252

str(centered_subsetted_x_test)
