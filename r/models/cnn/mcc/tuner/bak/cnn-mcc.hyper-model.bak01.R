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
    
    `__init__` = function(self, 
                          num_classes,
                          learning_rate) {
      
      self$num_classes = num_classes
      self$learning_rate = learning_rate
      NULL
    },
    
    build = function(self, hp) { # [2]
      put_log("Building next model for tuning with learning rate %1...",
              self$learning_rate)
      
      input_layer <- layer_input(shape = shape(28L, 28L, 1L))
      layer <- input_layer

      # conv_filters0 <- hp$Int('filters_0',
      #                        min_value = 32,
      #                        max_value = 256,
      #                        step = 32)
      
      # layer <- input_layer |>
      #   layer_conv_2d(filters = conv_filters0,
      #                 kernel_size = c(3L, 3L),
      #                 # padding = 'same',
      #                 # strides = list(1L, 1L),
      #                 activation = "relu") |>
      #   layer_max_pooling_2d()
      
      conv_blocks <- hp$Int('conv_blocks',
                         min_value = 1,
                         max_value = 5,
                         default = 3)
      
      put_log("Processin %1 Convolution Blocks...", conv_blocks)
      
      conv_filters <- hp$Int(paste0('filters_', i),
                             min_value = 32,
                             max_value = 256,
                             step = 32)
      
      for (i in 1:conv_blocks) {
        

          put_log("Processing the Convolution block %1 with filters %2...", 
                  i, conv_filters)

        
        put_log("Adding Convolution layer for Block %1
with filters %2...", i, conv_filters0)
        
        layer <- layer |>
          layer_conv_2d(filters = conv_filters*i,
                        kernel_size = c(3L, 3L),
                        # padding = 'same',
                        # strides = list(1L, 1L),
                        activation = "relu") |>
          layer_max_pooling_2d()
        
        put_log("Max pooling layer has been added to Block %1", i)
        
        
#         for (j in 1:2) {
#           # conv_filters <- 32L*j
#           
#           put_log("Adding Convolution layer %1 for Block %2
# with filters %3...", j, i, conv_filters)
#           
#           layer <- layer |>
#             layer_conv_2d(filters = conv_filters0,
#                           kernel_size = c(3L, 3L),
#                           # padding = 'same',
#                           # strides = list(1L, 1L),
#                           activation = "relu") |>
#             layer_max_pooling_2d()
#             put_log("Max pooling layer has been added to Block %1", i)
#           # layer <- layer |>
#           #   layer_max_pooling_2d(
#           #     # pool_size = list(2L, 2L)
#           #   )
#         }

      }

      # put_log("Adding Max pooling layer...")
      # layer <- layer |> layer_max_pooling_2d()
      # put_log("Max pooling layer has been added")
      
      put_log("Adding the final dense hidden layer...")
      
      layer <- layer |>
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

      put_log("Creating an output layer...")
      
      output_layer <- layer |>
        layer_dense(as.integer(self$num_classes),
                    activation = 'softmax')

      put_log("Compiling the next model being tuned...")
      
      model <- keras_model(input_layer, output_layer) |>
        compile(
          optimizer = keras3::optimizer_adamax(learning_rate = self$learning_rate),
          loss = 'sparse_categorical_crossentropy',
          metrics = 'accuracy')
      
      put_log("The next model for tuning has been compiled.")
      
      return(model)
    }
  )
)
