## Model Blending

Simply take all the models and average their predictions in some sensible way.

For regression, the simplest option would be the mean (or a weighted mean, if you want to give some models more influence than others)

For classification, use a (weighted) geometric mean

If asigning weights to these for some reason, careful not to over-fit.  For example weighting based on CV scores, or public leaderboard scores could do more harm than good.

### Majority Voting

For a classification problem, a simple method to blend an odd number of classifiers into a majority voting scheme.  This works well with the following characteristics:

- A Diverse collection of algorithms or training data
- Independent/uncorrelated predictions from each
- Use of individual knowledge for any or all of the classifiers

The individual models can all be fitted with the same data in a single step using `VotingClassifier` module from scikit-learn.

```python
# Instantiate the individual models
clf_knn = KNeighborsClassifier(5)
clf_lr = LogisticRegression(class_weight='balanced')
clf_dt = DecisionTreeClassifier(min_samples_leaf=3, min_samples_split=9, random_state=500)

# Create and fit the voting classifier
clf_vote = VotingClassifier(
estimators=[('knn', clf_knn), ('lr', clf_lr), ('dt', clf_dt)])
clf_vote.fit(X_train, y_train)
```

## Averaging (soft voting)
The syntax is the same, to make a voting classifier, just specify the voting parameter `voting='soft'`

For classification problems, the probabilities will be averaged
For regression problems, the mean prediction will be used.

## Bagging

This is a technique for use with 'weak' models.  By which we mean models that meet these criteria

- Performance better than random guessing, but not very much better
- Light weight
- Fast


## Model Stacking

Split the test data into `train_1` & `train_2`.  With `train_1` train a bunch of different models.  
```python
train_1, train_2 = train_test_split(train_ids, test_size=0.5, random_state=123)
```

With `train_2` use those model outputs as features for the new model.   For example (1) could be a CNN classification with a bunch of different networks.  (2) could use XGBoost or some other tree method to combine the predictions.  That's a pretty neat idea, I'm going to try it some time in a Kaggle comp.