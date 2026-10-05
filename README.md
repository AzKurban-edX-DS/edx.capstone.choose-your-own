
<!-- README.md is generated from README.Rmd. Please edit that file -->

# edx.Capstone Choose Your Own Project

This is the ***Capstone Choose Your Own*** Project as part of the [edX
HarvardX PH125.9x Data Science Capstone
Course](https://pll.harvard.edu/course/data-science-capstone) to build a
*Choose Your Own Project* according to the course requirements.

> [!IMPORTANT]
> This project is implemented as [RStudio Project](https://support.posit.co/hc/en-us/articles/200526207-Using-RStudio-Projects).

The goal of the project is to build a *Handwritten Character Recognition
System* using the [HandWritten_Character
Dataset](https://www.kaggle.com/datasets/vaibhao/handwritten-characters)
downloaded from the [Kaggle](https://www.kaggle.com) platform.

As described on the [landing
web-page](https://www.kaggle.com/datasets/vaibhao/handwritten-characters),
the *context* of the dataset is as follows:

1.  Used *EMNIST* data for Alphabets and Digits.
2.  Transformed the Data using some image processing techniques and
    convert it to 32,32 pixel black and white images.
3.  Created the Data set for special character ( @, \#, \$, & )
4.  Merged the Categories to avoid the misclassification
5.  Total 39 Categories in Train and Validation set

## Essential Directories and Files

The following are essential directories and files of the project:

- [edx.capstone.choose-your-own.Rproj](edx.capstone.choose-your-own.Rproj):
  The [RStudio
  Project](https://support.posit.co/hc/en-us/articles/200526207-Using-RStudio-Projects)
  file;
- [index.R](r/index.R): The *Index (main) `R` Script*;
- [support-functions](r/support-functions) folder: Directory for the
  *User-defined Functions* scripts used throughout the project;
- [support-scripts](r/support-scripts) folder: Directory for the
  *Auxiliary (Support) Scripts* used by the *Index `R` Script*;
- [setup.R](r/support-scripts/setup.R): Support script for installing
  all necessary packages, loading required libraries, and resolving
  conflicts;
- [prepare-input-data.R](r/support-scripts/prepare-input-data.R):
  Support script for preparing datasets for training and testing the
  *Handwritten Character Classifier* models developed in the project;
- [load-flattened-dataset.R](r/support-scripts/load-flattened-dataset.R):
  Support script for loading flattened dataset used by the following
  models:
  - ***kNN+PCA MCC***: *Multiclass Classifier (MCC)* based on the
    [k-Nearest Neighborhood
    (kNN)](https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/resampling-methods.html#sec-knn-cv-intro)
    algorithm, with the use of the [Principal Component Analysis
    (PCA)](https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/dimension-reduction.html)
    method for [preprocessing of
    predictors](https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/ml-in-practice.html#preprocessing);
  - ***RF MCC:*** *MCC* based on the [Random forests
    (RF)](https://rafalab.dfci.harvard.edu/dsbook-part-2/ml/algorithms.html#sec-random-forests)
    algorithm;
  - ***DNN MCC:*** *Basic [Deep
    Learning](https://www.geeksforgeeks.org/deep-learning/introduction-deep-learning/)
    (BDL) MCC*;
- [models](r/models) folder: Directory containing scripts for all the
  models used in the project;
- [cnn](r/models/cnn) folder: Directory containing scripts for the
  following [Convolutional Neural Network
  (CNN)](https://learnopencv.com/understanding-convolutional-neural-networks-cnn/)
  models:
  - ***CNN MCC:*** *CNN-based MCC*;
- [HW-Chars.Recognition.Site](reports/HW-Chars.Recognition.Site) folder:
  Directory for the *.RMD Report* files;
- `capstone.choose-your-own.report.pdf`: (Not implemented yet) The final
  *.PDF report* - the ultimate product of the project;
- ***data*** (local) folder: (automatically created if it does not
  exist, after the first project run) Directory for all the data files
  used in the project.

> [!IMPORTANT]
> The contents of the *data* folder are not tracked on *GitHub*.

## Working Environment

Since this project is an [RStudio
Project](https://support.posit.co/hc/en-us/articles/200526207-Using-RStudio-Projects),
it is recommended to work with the project in *RStudio IDE*. As
mentioned above in section [Essential Directories and
Files](#essential-directories-and-files), the *RStudio Project* file is
[edx.capstone.choose-your-own.Rproj](edx.capstone.choose-your-own.Rproj).

### Prerequisites & Setup

#### 1. R and RStudio

Ensure you have `R` (version **4.0** or higher recommended) and
**RStudio** installed.

> [!NOTE]
> The author completed this project using `R` version **4.6.1** and **RStudio 2026.09.0** *(Build 174)*.

#### 2. Environment Setup

- **On Windows OS:** Run the
  [\_\_setup-env.cmd](r/support-scripts/setup-env/__setup-env.cmd)
  script in PowerShell or Command Prompt;

> [!TIP]
>Ensure the `R bin folder` is added to your Windows System Environment Variables (`PATH`) so that `Rscript.exe` can be executed from any folder

> [!NOTE]
> The author used [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7.6) on Windows 11 OS.

- **On other Operating Systems:** Manually run the `R` scripts below in
  the specified order:

  - [1.install-reticulate&miniconda.R](r/support-scripts/setup-env/1.install-reticulate&miniconda.R);
  - [2.install-tensorflow.R](r/support-scripts/setup-env/2.install-tensorflow.R);
  - [3.install-keras3.R](r/support-scripts/setup-env/3.install-keras3.R);
  - [4,install-kerastuner.R](r/support-scripts/setup-env/4.install-kerastuner.R).

> [!IMPORTANT]
> If you are reinstalling the previously configured environment, run the [__uninstall-all.R](r/support-scripts/setup-env/__uninstall-all.R) script first to remove the packages previously installed by the script mentioned above.

## Running the Project
