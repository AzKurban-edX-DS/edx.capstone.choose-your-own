import keras
from keras import layers
import keras_tuner as kt

# ==========================================
# PHASE 1: TUNING THE ARCHITECTURE ONLY
# ==========================================

def build_architecture_model(hp):
    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs
    
    # Tune structural hyperparameters
    for i in range(hp.Int("num_conv_layers", 1, 3)):
        x = layers.Conv2D(
            filters=hp.Int(f"filters_{i}", min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(f"kernel_{i}", values=[3, 5]),
            activation="relu",
            padding="same"
        )(x)
        x = layers.MaxPooling2D()(x)
        
    x = layers.Flatten()(x)
    x = layers.Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1))(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Use a fixed, standard learning rate for structural phase
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Initialize the architecture tuner
arch_tuner = kt.Hyperband(
    build_architecture_model,
    objective="val_accuracy",
    max_epochs=10,
    factor=3,
    directory="cnn_tuning",
    project_name="phase1_architecture"
)

# Execute Phase 1 Search
# arch_tuner.search(x_train, y_train, validation_data=(x_val, y_val))

# Retrieve the best structural layout
best_hps_arch = arch_tuner.get_best_hyperparameters(num_trials=1)[0]


# ==========================================
# PHASE 2: TUNING THE OPTIMIZER & COMPILATION
# ==========================================

def build_optimization_model(hp):
    # 1. Rebuild the winning architecture from Phase 1
    # We pass the best_hps_arch values explicitly so they remain static
    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs
    
    for i in range(best_hps_arch.get("num_conv_layers")):
        x = layers.Conv2D(
            filters=best_hps_arch.get(f"filters_{i}"),
            kernel_size=best_hps_arch.get(f"kernel_{i}"),
            activation="relu",
            padding="same"
        )(x)
        x = layers.MaxPooling2D()(x)
        
    x = layers.Flatten()(x)
    x = layers.Dropout(best_hps_arch.get("dropout"))(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # 2. Tune optimization hyperparameters separately
    lr = hp.Float("learning_rate", min_value=1e-5, max_value=1e-2, sampling="log")
    optimizer_choice = hp.Choice("optimizer", values=["adam", "rmsprop"])
    
    if optimizer_choice == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Initialize the optimization tuner
opt_tuner = kt.Hyperband(
    build_optimization_model,
    objective="val_accuracy",
    max_epochs=15, # Often optimization tuning benefits from slightly longer schedules
    factor=3,
    directory="cnn_tuning",
    project_name="phase2_optimization"
)

# Execute Phase 2 Search
# opt_tuner.search(x_train, y_train, validation_data=(x_val, y_val))
