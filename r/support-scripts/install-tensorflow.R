.rs.restartR()
if(!require(reticulate)) {
  install.packages(c("reticulate", "tensorflow", "keras3"))
  
  
  library(reticulate)
  # install_python(list = TRUE)
  # install_python()
  # install_python(version = "3.10.11")
  # py_require()
  options(reticulate.conda_binary = "C:\\ProgramData\\anaconda3\\condabin\\conda.bat")
  
  # 1. Create a new Conda environment with a specific Python version
  # (e.g., creating an environment named "ml-env" with Python 3.10)
  conda_create(envname = "r-tensorflow_py3.10", python_version = "3.10")

  # 2. Tell reticulate to use this new environment for the rest of your session
  use_condaenv("r-tensorflow_py3.10", required = TRUE)
  #py_config()
  
  # 1. Install TensorFlow first
  tensorflow::install_tensorflow(method = "conda", envname = "r-tensorflow_py3.10")
  
}
