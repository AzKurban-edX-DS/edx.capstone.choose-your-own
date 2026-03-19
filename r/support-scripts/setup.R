# Setup -----------------------------------------------------------------------
options(timeout = max(300, getOption("timeout")))

#> Reference: Some ideas and code snippers were used from the following GitHub repository:
#> https://github.com/AzKurban-edX-DS/harvardx-movielens
if(!require("logr")) 
  install.packages("logr")

if(!require(tidyverse))
  install.packages("tidyverse")

library(tidyverse)

#> `stringr` library is already included to the `tidyverse` package,
#> there's no need to install `stringr`
# if(!require(stringr))
#   install.packages("stringr")

# if(!require())
#   install.packages("")
# if(!require())
#   install.packages("")

if(!require(remotes))
  install.packages("remotes")

# if(!require(tensorflow))
#   remotes::install_github("rstudio/tensorflow")

# if(!require(tensorflow))
#   remotes::install_github("rstudio/tensorflow")
# 
# library(tensorflow)
# install_tensorflow(envname = "r-tensorflow")


if(!require(reticulate))
  install.packages("reticulate")

library(reticulate)

# py_require(python_version = ">=3.11")
reticulate::install_python(version = "3.11")

if (!require("pacman")) 
  install.packages("pacman")

library(pacman)
p_load(conflicted)

if(!require(keras3)) {
  install.packages("keras3")
  install_keras()
}

library(tensorflow)
library(keras3)

conflict_prefer("shape", "keras3", quiet = TRUE)
conflict_prefer("set_random_seed", "keras3", quiet = TRUE)

# reticulate::virtualenv_remove("r-tensorflow")
# install_tensorflow(extra_packages="pillow")
# install_tensorflow(envname = "r-tensorflow")

tf$constant("Hello TensorFlow!")
tensorflow::tf_version()

