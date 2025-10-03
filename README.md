# MachineLearningLab7

# Ensemble Learning Analysis

This project explores **ensemble methods** (majority voting, bagging, AdaBoost, random forest) compared against individual classifiers (logistic regression, decision tree, KNN) on datasets like Wine and Iris.

---

## Majority Voting vs. Individual Classifiers

The majority voting classifier achieved a ROC AUC of 0.97 (±0.04), surpassing logistic regression (0.92 ±0.09), decision tree (0.88 ±0.07), and KNN (0.89 ±0.10). This edge comes from the ensemble’s ability to balance out individual model flaws. Each classifier—whether it’s logistic regression struggling with nonlinear patterns or KNN faltering on outliers—makes unique errors. Voting aggregates their predictions, smoothing out mistakes and reducing both variance and bias. However, it’s not always a win. If the base models are too similar (e.g., all decision trees) or one consistently misleads the group, the ensemble can underperform. Small datasets also limit diversity, making voting less effective.


## Bagging Analysis
Ramping up the number of estimators in bagging, from 100 to 500, lifted test accuracy from roughly 0.89 to 0.917. More trees stabilize predictions by averaging out fluctuations, though gains taper off around 200 estimators, with added computational cost. Bootstrap sampling—drawing random subsets with replacement—creates diverse training sets, allowing trees to learn slightly different patterns. This diversity cuts variance without inflating bias. A single decision tree overfits (train: 1.000, test: 0.833), but bagging (train: 1.000, test: 0.917) smooths predictions, making it more robust for complex, nonlinear data like the Wine dataset.

## AdaBoost Insights
AdaBoost’s learning rate shapes its performance. A lower rate (e.g., 0.1) slows convergence but promotes generalization by making cautious weight updates, while a higher rate (e.g., 1.0) speeds things up but risks overfitting to noise. In the error convergence plot, training error drops to near zero, and test error stabilizes at ~0.08 without rising, thanks to the conservative 0.1 rate avoiding overemphasis on outliers. Decision stumps work well as base learners because they’re intentionally weak (train accuracy ~0.916, close to random for binary classification). This simplicity lets AdaBoost iteratively focus on misclassified samples, building a strong ensemble step by step.
## Comparative Performance
On the Iris dataset, KNN led with a 0.9518 accuracy, likely because Iris’s low-dimensional, near-linearly separable structure suits distance-based methods. Ensembles like voting and bagging scored ~0.94, benefiting from their ability to combine diverse or stable predictions. Random Forest, a close relative of bagging, hit 0.9318 by adding random feature subsampling to reduce tree correlation. Voting is ideal for blending varied models (e.g., for structured tabular data), while bagging and AdaBoost excel with tree-based models on unstructured or noisy data. Random Forest is a go-to when automatic feature selection is needed, especially for high-dimensional datasets.

## Practical Considerations
Majority voting is computationally lightweight, as it parallelizes a small set of models. Bagging and AdaBoost, however, scale with the number of estimators (complexity ~ O(n_estimators * base_model_cost)), making them more demanding but GPU-friendly. Larger ensembles reduce variance, which is great for noisy data, but weak base learners can introduce bias if not carefully tuned—early stopping helps find the sweet spot. Voting suits quick prototyping (e.g., in data science competitions), bagging is effective for high-variance problems like financial forecasting, and AdaBoost shines in scenarios like medical diagnostics, where correcting rare errors is critical.

**When to use**:
- Voting → Fast prototyping (e.g., Kaggle competitions)  
- Bagging → Volatile data (e.g., stock prices)  
- AdaBoost → Error-sensitive tasks (e.g., rare disease detection)  

---

## Key Takeaway
Ensembles consistently outperform individual models by blending strengths:  
- Majority Voting → Best overall (ROC AUC 0.97)  
- Bagging → Stabilizes high-variance learners  
- AdaBoost → Focuses on hard-to-classify cases  
- Random Forest → Balanced, with feature randomness  

Ensembles cost more compute but deliver **robust, reliable results** when matched to dataset quirks.
