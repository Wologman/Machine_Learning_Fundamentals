## Model Blending

Simply take all the models and combine their predictions in some sensible way.

For regression, the simplest option would be the mean (or a weighted mean, if you want to give some models more influence than others)

For classification, use a (weighted) geometric mean

If asigning weights to these for some reason, careful not to over-fit.  For example with Kaggle comps, weighting based on CV scores, or public leaderboard scores could do more harm than good.

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

## Bagging (Bootstrap Agregating)

This is a technique for use with 'weak' models.  By which we mean models that meet these criteria

- Performance better than random guessing, but not very much better
- Light weight
- Fast

By combining a large enough number of weak models, we can potentially get a strong model.  Condorcet's Jury theorum is that if the following requirements are met, adding models will improve performance of the ensemble, and approach 1 (100%).  The requirements are:

- Models are independent
- Each model performs better than random guessing
- All individual models have similar performance

With previous examples we looked at *heterogenious* models.  For example combining KNN, LR & DT in one ensemble.  Bagging refers to aggregating *homogenious* weak models, by random sub-sampling of the data.  By doing this with suitable data we will also satisfy the conditions  above and get a strong model.  

For example, if we build weak (say max_depth=4) decision tree classifiers, sampling with replacement, and changed the random seed every time, then we could build a home made bagging classifier:

```python
def build_decision_tree(X_train, y_train, random_state=None)
	#Takes a sample with replacement,
    # builds a "weak" decision tree,
    # and fits it to the train set

def predict_voting(classifiers, X_test):
    # Makes the individual predictions 
    # and then combines them using "Voting"

clf_list = []
for i in range(21):
	weak_dt = build_decision_tree(X_train, y_train, i)
	clf_list.append(weak_dt)

# Predict on the test set
pred = predict_voting(clf_list, X_test)

# Print the F1 score
print('F1 score: {:.3f}'.format(f1_score(y_test, pred)))
```

For the scikit-learn implementation:
```python
# Instantiate the base model
clf_dt = DecisionTreeClassifier(max_depth=4)

# Build and train the Bagging classifier
clf_bag = BaggingClassifier(clf_dt, 21, random_state=500)
clf_bag.fit(X_train, y_train)

# Predict the labels of the test set
pred = clf_bag.predict(X_test)

# Show the F1-score
print('F1-Score: {:.3f}'.format(f1_score(y_test, pred)))
```
A the form is the same for bagging of other weak models.  Just intantiate a different base_estimator model.

We can also set a parameter `oob_score=True` and the bagging process will set aside some samples, then evaluate on those.  The metric will depend on the type of classifier, but for classification: accuracy, regression: R^2    The result can be accessed with `clf_bag.oob_score_`

### Bagging parameter choice

Play with these to get better models
- `n_estimators`: By default 10, but should be more.  Usually 100-500 trees.
- `max_samples`: Number of samples to draw for each estimator
- `max_features` Number of features to use for each estimator
		Classification ~ sqrt(number of features)
		Regression ~ number of features/3
- `bootstrap` Whether samples are drawn with replacement
		If `True` max_samples = 1 
		If `False` max samples < 0, otherwise all samples will be identical

### Random Forest
Random forest is just a special case of bagging. Using a large number of small decision trees.  Since scikit-learn provides a built in module, we might as well use this.

## Model Stacking

Split the test data into `train_1` & `train_2`.  With `train_1` train a bunch of different models.  
```python
train_1, train_2 = train_test_split(train_ids, test_size=0.5, random_state=123)
```

With `train_2` use those model outputs as features for the new model.   For example (1) could be a CNN classification with a bunch of different networks.  (2) could use XGBoost or some other tree method to combine the predictions.  That's a pretty neat idea, I'm going to try it some time in a Kaggle comp.