options(timeout = max(300, getOption("timeout")))

if(!require(pak)) 
  install.packages("pak")

if(!require(kerastuneR)) {
  #install.packages("kerastuneR")
  library(pak)
  pak::pak('eagerai/kerastuneR')
  
  library(kerastuneR)
  library(reticulate)
  
  # Tell reticulate to use this new environment for the rest of your session
  use_miniconda("mini.r-tensorflow_py3.11", required = TRUE)
  
  kerastuneR::install_kerastuner(method = "conda", 
                               envname = "mini.r-tensorflow_py3.11",
                               tensorflow = '2.21',
                               restart_session = FALSE)
  .rs.restartR()
}

