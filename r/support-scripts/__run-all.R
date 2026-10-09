support_scripts.dir <- "r/support-scripts"


## Run Project Scripts ---------------------------------------------------------
index.script_path <- "r/index.R"

stopifnot(file.exists(index.script_path))

source(index.script_path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

## Render PDF Reports ----------------------------------------------------------
rmd_render.script_path <- file.path(support_scripts.dir, "__rmd.render.R")
stopifnot(file.exists(rmd_render.script_path))

source(rmd_render.script_path,
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)
