#%%%%%%%%%%%%%%%%%%%%%
# Load Flatten Dataset
#%%%%%%%%%%%%%%%%%%%%%

### Open log: Load Input Data ----------------------------------___-------------
open_logfile(".load-input-data")
### Load Input Dataset ---------------------------------------------------------

start <- put_start_date()
put_log("Loading flatten training dataset from the backup file...")

x <- readRDS(my_emnist.file_path)

put_log("The flatten training dataset has been loaded from the following file:
%1", my_emnist.file_path)

put_log("The flatten dataset has the following structure:
  %1", capture.output(str(x)))

if(!exists("y.labels"))
  y.labels <- readRDS(classifier.label_list.file_path)

put_log("The Classifier Handwritten Character List contains the following labels:
%1", y.labels, .sep = " ")

y.groups <- ds.get_classIDs.grouped(x)
y <- y.groups$classID
str(y)
length(y)

y.int <- as.integer(y)
y.chars <- y.groups$groupByClass
str(y.chars)
### Close Log ------------------------------------------------------------------
log_close()
