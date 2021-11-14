# Classification-with-Wisconsin-Breast-Cancer-DataSet
##PHW1

##Introduction 
Breast cancer is the second leading cause of cancer death among women. And the incidence of breast cancer continues to increase every year. Therefore, improving diagnostic accuracy by analyzing breast cancer data is an important task. This analysis aims to find best model which classifies whether the breast cancer is benign or malignant. I applied various scalers, hyperparameters, and classifiers to derive the best combination model with the highest accuracy.

##Identify the Dataset
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

#  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)

- Number of Attributes: 10 and class attribute
Attributes 2 through 10 have been used to represent instances. Each instance has one of 2 possible classes: benign or malignant.

- Number of Instances: 699 
- Missing attribute values: 16
    There are 16 instances in Groups 1 to 6 that contain a single missing 
    (i.e., unavailable) attribute value, now denoted by "?".
