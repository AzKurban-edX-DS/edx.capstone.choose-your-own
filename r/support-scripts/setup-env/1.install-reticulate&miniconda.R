if(!require(reticulate)) {
  options(timeout = max(300, getOption("timeout")))
  install.packages("reticulate")
  
  library(reticulate)
  reticulate::install_miniconda()
}
