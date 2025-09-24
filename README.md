
# Classification of Chronic Kidney Disease Using Machine Learning Classifiers

## Overview

Chronic Kidney Disease (CKD) is a growing health concern worldwide, and often it is detected only in later stages. This project aims to detect CKD earlier by studying medical data and applying machine learning methods. Different classifiers were tested and compared to identify which ones give the most dependable results, with the goal of helping doctors and patients receive clearer, timely insights that support better treatment and care.

##  Objectives

* Detect CKD at early stages using patient medical records.
* Compare multiple machine learning classifiers for accuracy and reliability.
* Identify important medical features that influence CKD classification.
* Provide a system that can support doctors in making quicker, data-driven decisions.

##  Technologies & Tools

* *Programming Language:* Python 3.x
* *Libraries:* Pandas, NumPy, Scikit-learn, Matplotlib
* *Framework:* Flask (for simple web interface)
* *IDE:* PyCharm
* *Database:* MySQL (for storing records)
* *Dataset:* UCI Machine Learning Repository – CKD Dataset

##  Methods & Approach

1. *Data Collection & Preprocessing* – handled missing values, normalized features, and prepared data for analysis.
2. *Feature Selection* – identified key features such as age, haemoglobin, and WBC count.
3. *Model Training* – applied multiple classifiers including:

   * Random Forest
   * Support Vector Machine (SVM)
   * Decision Tree
   * K-Nearest Neighbors (KNN)
4. *Evaluation* – compared models on accuracy, sensitivity, and specificity.
5. *Result* – SVM achieved the best performance with *\~86.5% accuracy*.

##  Results

* Support Vector Machine outperformed other classifiers.
* Key features like haemoglobin levels, WBC count, and age had the most impact.
* Feature selection improved classifier accuracy by up to *4.7%*.

##  Future Enhancements

* Integration with live hospital data for real-time prediction.
* Deploying as a cloud-based application for wider access.
* Extending the model with deep learning techniques.
* Adding visualization dashboards for doctors and patients.

## Acknowledgements

This work was carried out as part of our *Final year B.Tech project* under the guidance of *Mr. S. Praveen Kumar , Assistant Professor , Department of CSE , DR M. G. R Educational and Research Institute,Chennai*. Special thanks to faculty, friends, and for their support.
