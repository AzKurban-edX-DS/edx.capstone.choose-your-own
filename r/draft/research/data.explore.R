## Explore imput datasets ------------------------------------------------------

str(img28x28bin.list$img.list$T$img.list)

lbl.img.list <- img28x28bin.list$img.list$T$img.list
str(lbl.img.list)

lbl.img.flat_ls <- lbl.img.list |> img_mx.list2flatten_matrix(label)
str(lbl.img.flat_ls)

image(lbl.img.flat_ls[1:400,])

