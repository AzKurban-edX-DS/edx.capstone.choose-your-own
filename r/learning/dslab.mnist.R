if(!require(dslabs))
  install.packages("dslabs")
if(!require(tidyverse))
  install.packages("tidyverse")


library(dslabs)
library(tidyverse)

mnist <- read_mnist()
y <- mnist$train$labels
str(mnist)
y

data(mnist_27)
str(mnist_27)


