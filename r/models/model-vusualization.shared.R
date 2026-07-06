#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Shared  Model Evaluation Results Visualization
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Evaluation Results: Visualization ------------------------------------------

put_log("Plotting ROC curves the Model Evaluation Results...")
CURRENT_MODEL.ROC.CURVES <- plot.ROC.curves(TEST_SET.TARGETS,
                                            EVAL.PREDICTED_PROBABILITIES)
Sys.sleep(3)

put_log("Plotting the Tuned BDL MCC Model Per-Class Accuracy...")
CURRENT_MODEL.ACCURACY_BY_CLASS <- plot.per_class.accuracy.bars(TEST_SET.TARGETS,
                                                                EVAL.PREDICTION_VALUES)
Sys.sleep(5)

# Confusion Matrix data suitable for Visualization using the `cvms` package:
# Reference: https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html

put_log("Creating a confusion matrix for Tuned BDL MCC Model in a format suitable for visualization 
using the `cvms` package...")
CURRENT_MODEL.CONFUSION_MATRIX <- create.confusion_matrix(TEST_SET.TARGETS,
                                                          EVAL.PREDICTION_VALUES)

put_log("Plotting the confusion matrix, please wait...")
plot.confusion_matrix(CURRENT_MODEL.CONFUSION_MATRIX)

log_close()








