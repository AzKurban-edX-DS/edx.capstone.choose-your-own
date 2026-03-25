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
  list(cimg.list = map_il(img_f$file_path.list, load.kaggle_img),
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
img_list$B$cimg.list

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

str(img.mx)
dim(img.mx)
class(img.mx)



# --------------------------------------------------

img.dat.probe <- create.hwChar_dataset(img.train.root_path, img.train.root_path)
str(img.dat.probe)



# --------------------------------------------------------

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








# library(sqldf)
# 
# img_set <- structure(img_list)
# 
# labels <- sqldf("select [label] from img_list")
# labels



# Reference:
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


