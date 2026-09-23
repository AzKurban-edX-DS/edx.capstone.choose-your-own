#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# `HyperParameters` Subclass for CNN MCC  Model Tuning
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kt <- import("keras_tuner")

CNN_MCC.HyperParameters <- reticulate::PyClass(
  'CNN_MCC.HyperParameters',
  inherit = kt$HyperParameters,
  defs = list(
    
    `__init__` = function(self, ...) {

      # self$hp = hp
      self$error = NULL
      
      super_obj <- py_eval("super")
      
      # Call Python parent __init__
      super_obj(CNN_MCC.HyperParameters, self)$`__init__`(...)
      NULL
    },
    
    set_fixed = function(self, hp) {
      
      hp_config <- hp$get_config()
      n.space <- length(hp_config$space)
      
      super_obj <- py_eval("super")
      
      for(i in 1:n.space) {
        hp_name <- hp_config$space[[i]]$config$name
        hp_vaue <- hp$values[[i]]
        
        super_obj(CNN_MCC.HyperParameters, self)$Fixed(hp_name,
                                                       value = hp_vaue)
      }
    }
  )
)
