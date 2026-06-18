if(!require(keras3)) {
  install.packages("keras3")
  # .rs.restartR()
  
  library(reticulate)
  options(reticulate.conda_binary = "C:\\ProgramData\\anaconda3\\condabin\\conda.bat")
  
  # 1. Create a new Conda environment with a specific Python version
  # (e.g., creating an environment named "ml-env" with Python 3.10)
  conda_create(envname = "r-tensorflow_py3.10", python_version = "3.10")
  
  # 2. Tell reticulate to use this new environment for the rest of your session
  use_condaenv("r-tensorflow_py3.10", required = TRUE)
  #py_config()
  
  # 2. Install Keras 3
  keras3::install_keras(method = "conda", envname = "r-tensorflow_py3.10")

}

