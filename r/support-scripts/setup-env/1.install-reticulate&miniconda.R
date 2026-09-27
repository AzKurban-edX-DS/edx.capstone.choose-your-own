if(!require(reticulate)) {
  options(timeout = max(300, getOption("timeout")))
  install.packages("reticulate", repos = "https://cloud.r-project.org")
  
  library(reticulate)
  reticulate::install_miniconda()
}
