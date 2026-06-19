#.rs.restartR()
if(!require(tensorflow)) {
  options(timeout = max(300, getOption("timeout")))
  install.packages("tensorflow")

  library(reticulate)

  # Create a new Conda environment with a specific Python version
  # (e.g., creating an environment named "ml-env" with Python 3.10)
  conda_create(envname = "mini.r-tensorflow_py3.11", python_version = "3.11")

  # Tell reticulate to use this new environment for the rest of your session
  use_miniconda("mini.r-tensorflow_py3.11", required = TRUE)

  # Install TensorFlow
  tensorflow::install_tensorflow(method = "conda", 
                                 envname = "mini.r-tensorflow_py3.11")
}
