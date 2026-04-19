# Train -------------------------
img.train.file_list <- img.file_path.get_list(img.train.root_path)
#names(img.train.file_list) <- train.labels
names(img.train.file_list)
str(img.train.file_list)


img.train.file_list.sample16 <- img.train.file_list[1:16]
str(img.train.file_list.sample16)

img.train.file_list15 <- img.train.file_list[[15]]

head(img.train.file_list15$file_path.list)
img.train.file_list15$label


img.train.file_list.sample16f7 <- lapply(img.train.file_list.sample16, function(ls){
  ls$file_path.list <- ls$file_path.list[1:7]
  ls
})

str(img.train.file_list.sample16f7)
head(img.train.file_list.sample16f7)

# train.img_list5 <- map_il(img.train.file_list15$file_path.list[1:5], imager::load.image)

# Probe -------------------

img.file_list.probe <- img.train.file_list.sample16f7
str(img.file_list.probe)
head(img.file_list.probe)
img.file_list.probe$A
img.file_list.probe$B$file_path.list

img_list <- lapply(img.file_list.probe, function(img_f){
  list(cimg.list = map_il(img_f$file_path.list, image_load.cimg),
       fpath.list = img_f$file_path.list)
})

str(img_list)
head(img_list)

length(img_list)
class(img_list)
names(img_list)
dim(img_list)

img_list$B
img_list$B$fpath.list
cimg.B1 <- img_list$B$char_matrix.list[[1]]
plot(cimg.B1)

img.B1 <- cimg.B1[,,1,1]
img.B1
vimg.B1 <- as.vector(img.B1)
vimg.B1
char.image(vimg.B1)

vimg.B1.bin <- (vimg.B1 > 0.1)*1
str(vimg.B1.bin)
vimg.B1.bin
plot(cimg.B1)
char.image(vimg.B1.bin)


img_hashtag1 <- img_list[[1]]$char_matrix.list[[1]]
img_hashtag1[,,1,1]

bin_img.B1


img_list[["B"]]
length(img_list[["B"]]$fpath.list)
img_list[["B"]]$cimg.list

class(img_list$B$cimg.list)
dim(img_list$B$cimg.list[[2]])

img_list$B$fpath.list
img_list.names <- names(img_list)
img_list.names
length(img_list.names)

img_file.named_list <- lapply(img_list.names, function(label){
  fpathslist <- img_list[[label]]$fpath.list
  fpaths <- as.vector(fpathslist)
  names(fpaths) <- base::rep(label, times = length(fpaths))
  fpaths

  # data.frame(
  #   label = base::rep(label, times = length(labeled_img.list)),
  #   file_path = labeled_img.list$fpath.list #,
  #   #img = labeled_img.list$cimg.list
  # )
})

str(img_file.named_list)

img_files <- unlist(img_file.named_list)
str(img_files)
dim(img_files)

img_list$B$cimg.list
dim(img_list$B$cimg.list)
length(img_list$B$cimg.list)
str(img_list$B$cimg.list)
# names(img_list$B$cimg.list)



cimg.list <- lapply(img_list.names, function(label){
  img_list[[label]]$cimg.list |> 
    as.matrix.cimg(label)
})

str(cimg.list)
# cimgs <- unlist(cimg.list)
# str(cimgs)
# dim(cimgs)

img.mx <- do.call(rbind, cimg.list)

char.image(img.mx[1,])

str(img.mx)
dim(img.mx)
class(img.mx)

  # img.mx[1:20, 1:20]
  char.image(img.mx[1,])
  str(cimg.list)
  str(cimg.list[[1]])
  char.image(cimg.list[[1]])
  char.image(cimg.list[[2]])
  char.image(cimg.list[[16]])
  char.image(cimg.list[[15]])
  dim(cimg.list[[15]])
  char.image(cimg.list[[15]][1,])
  dim(cimg.list[["A"]])
  names(cimg.list[[15]])
  rownames(cimg.list[[15]])
  img_A <- cimg.list[[15]]
  dim(img_A)
  char.image(img_A[1,])
  char.image(img_A[2,])
  char.image(img_A[3,])
  char.image(img_A[4,])
  dim(img_A[4,])
  img_A[4,]
  str(img_list)
  str(img_list["A"])
  str(img_list[["A"]])
  img_list.A <- img_list[["A"]]
  str(img_list.A)
  cimg_list.A <- img_list.A[[1]]
  str(cimg_list.A)
  str(cimg_list.A[[1]])
  plot(cimg_list.A[[1]])
  
  cimg.A_1 <- cimg_list.A[[1]]
  plot(cimg.A_1)
  
  mx.A_1 <- cimg.A_1[,,1,1]
  dim(mx.A_1)
  mx.A_1
  plot(mx.A_1)
  image(mx.A_1)
  
  mx.A_1.vector <- as.vector(mx.A_1)
  plot(mx.A_1.vector)
  char.image(mx.A_1.vector)
  
  chars.mx.list <- cimg.list
  str(chars.mx.list)
  rownames(cimg.list[[15]])
  
  cimg.list.A <- cimg.list[[15]]
  dim(cimg.list.A)
  
  cimg.list.A_1 <- cimg.list.A[1,]
  cimg.list.A_1
  char.image(cimg.list.A_1)
  
  label <- "A"
  plot(img_list[[label]]$cimg.list[[1]])
  
  cimg_list.A <- img_list[[label]]$cimg.list
  cimg_list.A_1 <- img_list[[label]]$cimg.list[[1]]
  plot(cimg_list.A_1)
  
# Shuffle Matrix rows ----------------------------------------------------------
  
  # 1. Create an example matrix
  mat <- matrix(data = 1:20, nrow = 5, byrow = TRUE)
  print("Original Matrix:")
  print(mat)
  
  # 2. Generate a random permutation of the row indices
  # nrow(mat) gives the total number of rows (in this case, 5)
  # sample() returns a random permutation of numbers from 1 to 5
  random_indices <- sample(nrow(mat))
  print("Random Row Indices:")
  print(random_indices)
  
  # 3. Use the random indices to reorder the matrix rows
  # The comma after 'random_indices' ensures all columns are selected
  shuffled_mat <- mat[random_indices, ]
  print("Shuffled Matrix:")
  print(shuffled_mat)
  
# Reserch:A_list.mx ---------------------
str(A_list.mx)  
dim(A_list.mx)  
str(A_list.mx[1,])  
char.image(A_list.mx[1,])  
char.image(A_list.mx[2,])  
  
# Research: Visualization ------------------------------------------------------

img.dat.probe <- hwChar_data.load(img.train.root_path, img_list.names)
str(img.dat.probe)
my_emnist.16ch <- img.dat.probe$my_emnist
char.image(my_emnist.16ch[1,])
char.image(my_emnist.16ch[15700,])
char.image(my_emnist.16ch[15701,])
char.image(my_emnist.16ch[15702,])


# Research: Arrays & Matrices --------------------------------------------------

length(images.df.list)
str(images.df.list)
dim(images.df.list)

img.list.mx <- t(images.df.list)
img.list.mx
str(img.list.mx)

unlist(img.list.mx)


img.array <- sapply(img_list, function(item){
  as.vector(item$cimg.list[,,1,1])
})

dim(img.array)
str(img.array)

### Open log: Load Train Data Subset (Max 64 files per char class) -------------
open_logfile(".load-train-data-subset64")
### Load Train Data Subset (Max 64 files per char class) -----------------------
char_files.max64 <- 64 
char_files.max64


ds.train.subset64.file_path <- file.path(ds.subsets.path, "train-data-subset64.RData")
ds.train.subset64.file_path

if (file.exists(ds.train.subset64.file_path)) {
  start <- put_start_date()
  load(ds.train.subset64.file_path)
  put_end_date(start)
} else {
  # Create Train Data list
  train.dat.subset64 <- hwChar_data.load(img.train.root_path, 
                                         char_files.max = char_files.max64)
  
  start <- put_start_date()
  save(train.dat.subset64, file = ds.train.subset64.file_path)
  put_end_date(start)
}

str(train.dat.subset64)

### Close Log ---------------------------------------------------------------
log_close()

# References: ------------------------------------------------------------------
#> 21  Working with Matrices in R
#> https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html 
#> 21.2 Case study: MNIST
#> https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/matrices-in-R.html#sec-mnist


# ----------------------------------


train.img_list5
str(train.img_list5)
train.img_list5
dim(train.img_list5)

train.list_0.1_5 <- list(images = train.img_list5,
                         file_list = img.train.file_list15$file_path.list[1:5])
str(train.list_0.1_5)

max(train.img_list5[[5]])
mean(train.img_list5[[1]])
mean(train.img_list5[[2]])
mean(train.img_list5[[3]])
mean(train.img_list5[[4]])
mean(train.img_list5[[5]])

plot(train.img_list5[[1]])

noise <- array(runif(5*5*5*3),c(5,5,5,3)) #5x5 pixels, 5 frames, 3 colours. All noise
dim(noise)

noise

noise.cimg <- as.cimg(noise)




img5list <- list()

for (i in seq_along(train.img_list5)) {
  img32x32_i <- train.img_list5[[i]][,,1,1]
  str(img32x32_i)
  img5list[[i]] <- img32x32_i
}

str(img5list)

img5array <- as.array(img5list)
str(img5array)


train.img_array5 <- as.array(train.img_list5)
dim(train.img_array5)
str(train.img_array5)

train.img5 <- train.img_list5[[1]]
train.img5
str(train.img5)
train.img5[,,1,1]
dim(train.img5)

train.img5.2d <- train.img5[,,1,1]
class(train.img5.2d)
dim(train.img5.2d)
dim(train.img_list5)
dim(as.array(train.img_list5))
dim(train.img5)
train.img_set5 <- train.img_list5[[,]]

train.img2d_list5 <- lapply(train.img_list5, function(train.img){
  train.img[,,1,1]
})

dim(train.img2d_list5)
str(train.img2d_list5)

train.img2d_array5 <- as.array(train.img2d_list5)
dim(train.img2d_array5)
str(train.img2d_array5)


