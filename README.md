Understand the data: Analyze and visualize the data to gain insights and understand the relationships between features and labels. Look for any missing values, outliers, or other inconsistencies that need to be addressed.

Feature engineering: Explore techniques to create new features or transform existing ones to improve model performance. This may include dimensionality reduction (e.g., PCA), feature scaling, or feature interaction.

Model selection: Identify suitable models to be tested, considering both linear and non-linear models, such as Ridge Regression, Lasso Regression, Support Vector Machines, Random Forest, XGBoost, and Neural Networks. Remember that the goal is to outperform the benchmark OLS model.

Model evaluation: Use cross-validation techniques to evaluate the performance of the selected models on the train/valid dataset. Compare their performance using the defined metrics: IC, Average Returns of Top 10%, and Accuracy of the Top 10%.

Hyperparameter tuning: Perform grid search or random search to optimize hyperparameters for each model, aiming to maximize performance on the train/valid dataset.

Ensemble techniques: Investigate whether stacking or blending models can improve performance. This may involve combining predictions from multiple models, either with simple averaging or by training a meta-model.

Test set evaluation: After selecting the best model or ensemble, evaluate its performance on the 2021 out-of-sample test dataset. Compare the results with the benchmark model and provide insights into the improvements achieved.

Iterative improvement: Continuously revisit models and ideas to further improve performance on the defined metrics. Explore alternative model architectures, feature engineering techniques, or training approaches.



test set ic: 0.07056382000848331

- 训练过程都按照rmse或mse等作为损失函数，oos metrics 
    1. 按照横截面算 IC Q9等指标，之后再算这些指标的均值 
    2. 直接在全部样本上算 IC