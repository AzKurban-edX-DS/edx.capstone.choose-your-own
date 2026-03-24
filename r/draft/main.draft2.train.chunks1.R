# Train -------------------------

fileList4 <- train_img32.file_list[[4]]
fileList1_4 <- train_img32.file_list[1:4]
str(fileList1_4)

head(fileList4$file_path.list)
fileList4$label

fileList5 <- train_img32.file_list[[5]]

head(fileList5$file_path.list)
fileList5$label

fileList5$label
train.img_list5 <- map_il(fileList5$file_path.list[1:5], imager::load.image)

train.img_list5
str(train.img_list5)
train.img_list5
dim(train.img_list5)

train.list_0.1_5 <- list(images = train.img_list5,
                         file_list = fileList5$file_path.list[1:5])
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


# Test --------------------------------------test.fileList4 <- test_img32.file_list[[4]]

head(test.fileList4$file_path.list)
test.fileList4$label

test.fileList5 <- test_img32.file_list[[5]]

head(test.fileList5$file_path.list)
test.fileList5$label
test.fileList5$dir_path

test.imgList5 <- imager::load.dir(test.fileList5$dir_path)
test.imgList5
#str(test.imgList5)
head(test.imgList5)
str(head(test.imgList5))
#-------------------------------------------------------------------------------
