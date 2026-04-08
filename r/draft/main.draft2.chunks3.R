char_files.max <- 64 
char_files.max

# Train Set ------------------

img.train.files <- img.train.dat$img.files
str(img.train.files)

train.label.list <- as.factor(names(img.train.files))
str(train.label.list)
train.label.list

img.train.list <- img.train.dat$img.list
str(img.train.list)
my_emnist.train <- img.train.dat$my_emnist

# Visualize the first char:
char.image(my_emnist.train[1,])

# Final Test Set ----------------------------------

