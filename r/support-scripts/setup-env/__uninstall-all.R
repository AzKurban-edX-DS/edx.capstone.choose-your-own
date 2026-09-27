if(require(keras)) {
  #library(keras)
  #detach("package:keras", unload = TRUE)
  remove.packages("keras")
}

if(require(kerastuneR)) {
  # detach("package:kerastuneR", unload = TRUE)
  remove.packages("kerastuneR")
}

if(require(autokeras)) {
  # detach("package:autokeras", unload = TRUE)
  remove.packages("autokeras")
}

if(require(keras3)) {
  # detach("package:keras3", unload = TRUE)
  remove.packages("keras3")
}

if(require(tensorflow)) {
  # detach("package:tensorflow", unload = TRUE)
  remove.packages("tensorflow")
}

if(require(reticulate)) {
  library(reticulate)
  print("Hello reticulate!")
  reticulate::miniconda_uninstall()
  detach("package:reticulate", unload = TRUE)
  remove.packages("reticulate")
}

# detach("package:", unload = TRUE)

.rs.restartR()

