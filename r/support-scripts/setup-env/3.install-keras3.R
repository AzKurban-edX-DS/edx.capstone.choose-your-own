if(!require(keras3)) {
  options(timeout = max(300, getOption("timeout")))
  install.packages("keras3")

  library(reticulate)

  # Tell reticulate to use this new environment for the rest of your session
  use_condaenv("mini.r-tensorflow_py3.11", 
               required = TRUE)

  # Install Keras 3
  keras3::install_keras(method = "conda", 
                        envname = "mini.r-tensorflow_py3.11")
}

