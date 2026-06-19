if(require(keras)) {
  detach("package:keras", unload = TRUE)
}

if(require(kerastuneR)) {
  detach("package:kerastuneR", unload = TRUE)
}

if(require(autokeras)) {
  detach("package:autokeras", unload = TRUE)
}

if(require(keras3)) {
  detach("package:keras3", unload = TRUE)
}

if(require(tensorflow)) {
  detach("package:tensorflow", unload = TRUE)
}

if(require(reticulate)) {
  library(reticulate)
  # print("Hello reticulate!")
  reticulate::miniconda_uninstall()
  detach("package:reticulate", unload = TRUE)
}

# detach("package:", unload = TRUE)

.rs.restartR()

