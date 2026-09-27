#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fine-Tuned CNN MCC Final Model: Evaluation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Setup -----------------------------------------------------------------------
open_logfile(".tuner.cnn-mcc.final-model.evaluation")
stopifnot(file.exists(tcnn_mcc.final_test.proc.path,
                      tcnn_mcc.final.file,
                      ds28x28.split.train_0.8.backup.file))

### Preparing a Final Test Set for the Final Model Evaluation ------------------

put_log("Loading the Test Set of 28x28x1-shape image data...")
test_set <- load28x28x1.test_set(ds28x28.split.train_0.8.backup.file)
put_log("The Training Set of 28x28x1-shape image data has been loaded from the following file:
%1", ds28x28.split.train_0.8.backup.file)

x_test <- test_set$x
str(x_test)
dim(x_test)

x_test.files <- test_set$files

y.test.groups <- test_set$class_groups

stopifnot(sum(as.character(y.test.groups$classID) != rownames(x_test)) == 0)

eval.targets <- y.test.groups$classID

y_test <- as.array(as.integer(eval.targets) - 1)
str(y_test)
dim(y_test)

#### Size of the Test Set by Class ------------------------------------------

put_log("The Training Set is balanced by the set of Classes:
%1", capture.output(print(y.test.groups$groupByClass, n = N.classes)))
{
  # A tibble: 39 × 2
  #    classID     n
  #    <fct>   <int>
  #  1 #         852
  #  2 $         852
  #  3 &         852
  #  4 @         852
  #  5 0         852
  #  6 1         852
  #  7 2         852
  #  8 3         852
  #  9 4         852
  # 10 5         852
  # 11 6         852
  # 12 7         852
  # 13 8         852
  # 14 9         852
  # 15 A         852
  # 16 B         852
  # 17 C         852
  # 18 D         852
  # 19 E         852
  # 20 F         852
  # 21 G         852
  # 22 H         852
  # 23 I         852
  # 24 J         852
  # 25 K         852
  # 26 L         852
  # 27 M         852
  # 28 N         852
  # 29 P         852
  # 30 Q         852
  # 31 R         852
  # 32 S         852
  # 33 T         852
  # 34 U         852
  # 35 V         852
  # 36 W         852
  # 37 X         852
  # 38 Y         852
  # 39 Z         852
  invisible()
}

rm(test_set)

## Running the Test Procedure --------------------------------------------------

source(tcnn_mcc.final_test.proc.path, 
       catch.aborts = TRUE,
       echo = TRUE,
       spaced = TRUE,
       verbose = TRUE,
       keep.source = TRUE)

