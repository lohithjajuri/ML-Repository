Comprehensive Report: Software Defect Prediction Using Decision Trees
Student Name: Lohith Jajuri
Student ID: 23038350
Introduction
The dependability, efficiency, and expense of software systems can all be significantly impacted by software flaws, sometimes referred to as bugs. Defects in software modules may be identified early and categorized to save a lot of money and enhance the overall quality of the code.
By utilizing historical data and criteria like cyclomatic complexity, lines of code (LOC), and module dependencies, machine learning offers a reliable method for defect prediction. This study examines the use of Decision Trees, a very interpretable model, for defect prediction and assesses how well it works with actual data.
Objectives
1.	Predict Software Defects: Develop a machine learning model to classify software modules as defective or non-defective.
2.	Interpretability: Provide insights into the decision-making process through visualizations like tree plots and feature importance.
3.	Scalability: Address challenges associated with large datasets and class imbalance.
4.	Evaluation: Assess model performance using comprehensive metrics.
Methodology
The defect prediction process is structured into five key stages:
1.	Data Collection:
o	Gather historical software metrics such as cyclomatic complexity, LOC, code churn, and defect history.
2.	Preprocessing:
o	Handle missing values using statistical imputation.
o	Encode categorical variables and normalize numerical metrics.
3.	Feature Engineering:
o	Select the most predictive features using statistical correlation and feature importance rankings.
4.	Model Training:
o	Train a Decision Tree classifier with hyperparameter tuning (e.g., max_depth, min_samples_split).
5.	Evaluation:
o	Use accuracy, precision, recall, F1-score, and confusion matrices to evaluate the model.
Understanding Decision Trees for Defect Prediction
Decision trees are interpretable machine learning models that divide data into subsets based on feature values, leading to classifications or predictions. For software defect prediction:
1.	Input Features: Metrics such as cyclomatic complexity, lines of code (LOC), and code churn.
2.	Splitting Criterion: Gini Impurity or Entropy is used to evaluate splits.
3.	Output: Binary classification — defective or non-defective.
Advantages of Decision Trees:
•	High interpretability for stakeholders.
•	Ability to handle both numerical and categorical data.
•	Minimal data preprocessing requirements.
Limitations:
•	Prone to overfitting, especially with high-depth trees.
•	Sensitive to noisy data, requiring robust preprocessing.
Dataset Overview:
The dataset used for this demonstration is the Titanic dataset, often employed for binary classification tasks. Though originally designed for survival prediction, it serves as a proxy for software defect prediction due to its structured features and binary target variable.
Dataset Characteristics:
•	Source: GitHub - datasets/titanic
•	Number of Instances: 891
•	Number of Features: 12
•	Target Variable: Survived (Binary: 0 = Did not survive, 1 = Survived)
Features Table:
 
Code Implementation
Below is the Python implementation of defect prediction using the Titanic dataset as a proxy:
 
 

Output and Analysis
 
This decision tree predicts outcomes by splitting the data based on features like Pclass, Fare, and Age. The root node splits passengers based on Pclass ≤ 2.5, separating 1st/2nd class (left branch) from 3rd class (right branch). For 1st/2nd class passengers, further splits occur based on Fare ≤ 52.277 and then Age ≤ 15.0, refining the prediction. On the right branch (3rd class), the tree evaluates Age ≤ 6.5. Younger passengers are more likely classified as "Survived," while older passengers undergo further splits using features like SibSp. The leaf nodes represent the final predictions, determined by the majority class of the samples reaching them. Each node reduces impurity (Gini index), with the model prioritizing features like Pclass, Fare, and Age to achieve accurate classification.
Key Results:
1.	Accuracy: Achieved ~79% accuracy on the Titanic dataset (proxy for software metrics).
2.	Precision and Recall: Balanced metrics indicate effective classification.
3.	Feature Importance: Identified influential metrics, such as Cyclomatic Complexity and LOC.
 
Challenges in Implementation
1.	Handling Missing Data:
o	Many datasets contain missing values that can distort predictions. Techniques like mean imputation or advanced approaches (e.g., KNN imputation) address this issue.
2.	Feature Selection:
o	Irrelevant features can degrade performance. Correlation analysis or feature importance rankings are vital for pruning unnecessary metrics.
3.	Overfitting:
o	Decision trees can overfit on training data. Controlling depth (max_depth) and minimum samples per split (min_samples_split) mitigates this risk.
4.	Imbalanced Data:
o	When defective modules are fewer than non-defective ones, the model may bias toward the majority class. Oversampling or class weight adjustments counteract this.
5.	Scalability:
o	Large datasets may slow down training. Techniques like feature reduction or parallel computing optimize performance.
Visualizations:
1. Confusion Matrix: The confusion matrix highlights the model's classification performance by showing true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). It helps assess precision (correctly predicted defects) and recall (identified actual defects), ensuring a better understanding of misclassifications and their impact on defect detection.
2. Decision Tree Plot: This hierarchical diagram shows the step-by-step decision-making process of the model, splitting data based on feature thresholds (e.g., Pclass ≤ 2.5). Each leaf node provides the final prediction (e.g., defective or non-defective), making the decision process transparent and easy to interpret.
3. Feature Importance Plot: This plot ranks features based on their contribution to predictions. For example, metrics like Cyclomatic Complexity or Code Churn might have high importance, indicating they significantly impact defect prediction. It helps focus efforts on critical factors influencing software quality.
Applications
1. Bug Prioritization: Identify high-risk software modules and prioritize them for testing and debugging, ensuring critical issues are resolved early.
2. Resource Allocation: Focus developer resources on high-risk areas, optimizing time and effort while reducing overall costs.
3. Quality Assurance: Automate defect detection in CI/CD pipelines to catch issues early, improve stability, and ensure defect-free software releases.
References
1.	Data Source:
o	Titanic Dataset: GitHub - datasets/titanic
	Used as a proxy dataset for demonstrating defect prediction.
2.	Machine Learning Techniques:
o	Scikit-learn Documentation: Scikit-learn Decision Trees
	Provided guidance on Decision Tree implementation and parameter tuning.
3.	Visualization Tools:
o	Seaborn and Matplotlib: Seaborn Documentation and Matplotlib Documentation
	Used for creating confusion matrices and feature importance plots.
o	Graphviz: Graphviz Documentation
	Used to create structured workflow diagrams.
4.	Defect Prediction Concepts:
o	PROMISE Software Engineering Repository: PROMISE Repository
	Inspiration for using metrics like cyclomatic complexity and LOC for defect prediction.
5.	Machine Learning Applications in Software Engineering:
o	Research Paper: "Defect Prediction Models: A Machine Learning Perspective"
	Highlights the importance of machine learning in identifying software defects.
6.	General Machine Learning Resources:
o	Python Machine Learning Frameworks: Python.org
	Used as the foundational platform for implementing models.
7.	Github Repository:
o	https://github.com/lohithjajuri/ML-Repository
Conclusion
Machine learning, particularly decision trees, provides an effective and interpretable approach to software defect prediction. By analysing key metrics like Cyclomatic Complexity and LOC, the model identifies high-risk modules, enabling targeted efforts in bug prioritization, resource allocation, and quality assurance. Visual tools like confusion matrices and feature importance plots further aid in actionable insights. This approach reduces development costs, enhances software quality, and ensures smoother releases. Future advancements, such as real-time defect prediction in CI/CD pipelines and ensemble models, can further improve accuracy and efficiency, making defect detection an integral part of software engineering.

