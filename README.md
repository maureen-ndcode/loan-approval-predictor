# loan-approval-predictor
#  AI Loan Approval Prediction using Ensemble Learning

This project predicts whether a loan application is likely to be approved based on an applicant's financial and credit information. More importantly, it was my way of understanding how different machine learning algorithms perform on the same problem and why ensemble methods are so powerful.

##  What I Learned

This project helped me understand the strengths and limitations of different classification algorithms.

* **Logistic Regression** served as a simple baseline model and showed that linear relationships alone can capture a good portion of the data.
* **Decision Tree** improved interpretability but was prone to overfitting.
* **Random Forest** demonstrated how bagging combines multiple trees to improve stability and reduce variance.
* **AdaBoost** showed how boosting focuses on correcting mistakes made by previous weak learners.
* **Gradient Boosting** built stronger models by sequentially minimizing prediction errors.
* **XGBoost** outperformed all other models by combining gradient boosting with optimizations such as regularization, tree pruning, and efficient computation.

This comparison helped me understand why XGBoost is widely used in real-world machine learning competitions and industry applications.

##  Model Performance

| Model               | Accuracy   |
| ------------------- | ---------- |
| Logistic Regression | **87.14%** |
| Decision Tree       | **87.22%** |
| Random Forest       | **91.44%** |
| AdaBoost            | **90.91%** |
| Gradient Boosting   | **92.04%** |
| ⭐ XGBoost           | **93.38%** |

XGBoost achieved the highest accuracy and was selected as the final model for deployment.

##  Tech Stack

* Python
* Pandas
* Scikit-learn
* XGBoost
* Streamlit
* Joblib

##  Deployment

The trained XGBoost model was deployed using Streamlit, allowing users to enter applicant details and receive a real-time prediction of loan eligibility.

---

*"Building this project helped me move beyond simply training models—I gained a practical understanding of ensemble learning, model comparison, preprocessing pipelines, and deploying machine learning models as interactive applications."*
