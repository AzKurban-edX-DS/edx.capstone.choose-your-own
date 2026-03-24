# Test --------------------------------------
test.fileList4 <- test_img32.file_list[[4]]

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
