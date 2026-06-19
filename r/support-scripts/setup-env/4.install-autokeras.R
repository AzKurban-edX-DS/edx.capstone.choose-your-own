if(!require(autokeras)) {
  options(timeout = max(300, getOption("timeout")))
  install.packages("autokeras")

  library(autokeras)
  library(reticulate)
  
  # Tell reticulate to use this new environment for the rest of your session
  use_miniconda("mini.r-tensorflow_py3.11", required = TRUE)
  
  autokeras::install_autokeras(method = "conda", 
                               envname = "mini.r-tensorflow_py3.11",
                               tensorflow = '2.21')
  
}

