#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# `HyperModel` Subclass for CNN MCC  Model Tuning
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#> `KerasTuner` `HyperModel` subclass for Convolutional Neural Network 
#> Multiclass Classifier (BDL MCC)  Model Tuning

# References:
# [1] R interface to Keras Tuner
# https://eagerai.github.io/kerastuneR/#r-interface-to-keras-tuner [1]

# [2] Hyperparameter tuning with Keras Tuner
# https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html

# [3] Easy Hyperparameter Tuning with Keras Tuner and TensorFlow
# https://pyimagesearch.com/2021/06/07/easy-hyperparameter-tuning-with-keras-tuner-and-tensorflow/

# library(keras3)
# library(tensorflow)
# library(magrittr)
# library(kerastuneR)

CNN_MCC.HyperModel <- reticulate::PyClass(
  'HyperModel',
  inherit = kerastuneR::HyperModel_class(),
  list(
    
    `__init__` = function(self, num_classes) {
      
      self$num_classes = num_classes
      NULL
    },
    
    build = function(self, hp) { # [2]
      input_layer <- layer_input(shape = shape(28L, 28L, 1L))
      
      #layer <- input_layer #|>
        # layer_conv_2d(filters = 32L,
        #               kernel_size = c(3L, 3L), 
        #               # strides = list(1L, 1L),
        #               activation = "relu") |>
        # # layer_max_pooling_2d() |>
        # layer_max_pooling_2d(pool_size = c(2, 2)) |>
        # # layer_dropout(rate = 0.25) |>
        # layer_conv_2d(filters = 64L, 
        #               kernel_size = c(3L, 3L),
        #               # strides = list(1L, 1L),
        #               activation = "relu") |>
        # # layer_max_pooling_2d() |>
        # layer_max_pooling_2d(pool_size = c(2L, 2L)) # |>
        #layer_dropout(rate = 0.25) |>
        # layer_flatten() |>
        # layer_dense(units = 128L, activation = "relu") |>
        # layer_dropout(rate = 0.5) #|>
        #layer_dense(units = N.classes, activation = "softmax")


      # for (i in 1:2) {

      for (i in 1:hp$Int('conv_blocs',
                         min_value = 2,
                         max_value = 5,
                         default = 3)) {

        conv_filters <- hp$Int(paste0('filters_', i),
                               min_value = 32,
                               max_value = 256,
                               step = 32)

        for (j in 1:2) {
          # conv_filters <- 32L*j
          
          layer <- input_layer |>
            layer_conv_2d(filters = conv_filters,
                          kernel_size = c(3L, 3L),
                          # padding = 'same',
                          # strides = list(1L, 1L),
                          activation = "relu")

          layer <- layer |>
            layer_max_pooling_2d()
          
          # if (hp$Choice(paste0('pooling_', i),
          #               values = c('avg', 'max')) == 'max') {
          #   layer <- layer |>
          #     layer_max_pooling_2d()
          # 
          # } else {
          #   layer <- layer |>
          #     layer_average_pooling_2d(pool_size = c(2L, 2L))
          # }

        }

      }

      layer <- layer |>
        # layer_global_average_pooling_2d() |>
        layer_dropout(hp$Float('dropout1',
                               min_value = 0.1,
                               max_value = 0.5,
                               step = 0.1,
                               default = 0.5)) |>
        layer_flatten() |>
        layer_dense(hp$Int('hidden_size',
                           min_value = 128L,
                           max_value = 512L,
                           step = 16,
                           default = 256L),
                    activation = 'relu') |>
        layer_dropout(hp$Float('dropout2',
                               min_value = 0.1,
                               max_value = 0.5,
                               step = 0.1,
                               default = 0.5))

      output_layer <- layer |>
        layer_dense(as.integer(self$num_classes),
                    activation = 'softmax')

      model <- keras_model(input_layer, output_layer) |>
        compile(
          optimizer = keras3::optimizer_adamax(
            hp$Float('learning_rate',
                     min_value = 1e-4,
                     max_value = 1e-2,
                     sampling = 'log')),
          loss = 'sparse_categorical_crossentropy',
          metrics = 'accuracy')
      
      return(model)
    }
  )
)
