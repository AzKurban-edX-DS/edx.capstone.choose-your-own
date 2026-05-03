# Fix structure for 2d CNN
train_array %
layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = "sigmoid")
summary(model)
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = "adam",
  metrics = c('accuracy')
)
history % fit(
  x = train_array, y = as.numeric(trainData$y),
  epochs = 30, batch_size = 100,
  validation_split = 0.2
)
