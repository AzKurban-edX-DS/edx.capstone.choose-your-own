if(!require(autokeras)) {
  install.packages("autokeras")
  # .rs.restartR()
  
  library(reticulate)
  options(reticulate.conda_binary = "C:\\ProgramData\\anaconda3\\condabin\\conda.bat")
  
    # 1. Create a new Conda environment with a specific Python version
  # (e.g., creating an environment named "ml-env" with Python 3.10)
  conda_create(envname = "r-tensorflow_py3.11", python_version = "3.11")
  
  # 2. Tell reticulate to use this new environment for the rest of your session
  use_condaenv("r-tensorflow_py3.11", required = TRUE)
  #py_config()
  py_require(python_version = "3.11")
  py_require()
  
  # library(tensorflow)
  
  # 1. Install TensorFlow first
  tensorflow::install_tensorflow(method = "conda", 
                                 envname = "r-tensorflow_py3.11", 
                                 python_version = "3.11")
  
  py_require(python_version = "3.11")
  py_require()
  
  tensorflow::tf_config()
  tensorflow::tf_version()
  
  .rs.restartR()
  
  library(reticulate)
  options(reticulate.conda_binary = "C:\\ProgramData\\anaconda3\\condabin\\conda.bat")
  use_condaenv("r-tensorflow_py3.11", required = TRUE)
  
  
  conda_list()
  
  # use_condaenv("r-tensorflow_py3.11", required = TRUE)
  #py_config()
  
    # 3. Finally, install AutoKeras
  library(autokeras)
  autokeras::install_autokeras(method = "conda", 
                               envname = "r-tensorflow_py3.11",
                               tensorflow = '2.21')
  
}

