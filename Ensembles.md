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
The form is the same for bagging of other weak models.  Just instantiate a different `base_estimator` model.

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

## Gradual Learning (Boosting)
Bagging was using collective learning.  With gradual learning we iteratively improve weak models by learning from the errors of previous ones.  Differenced & similarities are summarised as follows:

|  Collective | Gradual   |
|---|---|
|  Wistom of the crowd |  Iterative learning |
| Independent estimators  | Dependent estimators  |
| Learning the same task for the same goal | Learning different tasks for the same goal |
|  Parallel building |  sequential building |
| Weak base estimators  | Weak base estimators |

With gradual leaning we need to be careful to stop learning before the model starts fitting to noise in the data.  We can try to do this by looking at the characteristics of the errrors, and see if the variance starts to display white noise.

- Errors uncorrelated with input features
- Errors are unbiased and have consistant variance.

Another stopping metric, a bit simpler to implemtnt could be a measure of improvement between iterationis.  Ie if *performance change* < *improvement threshold* then stop leraning.

Here is an example of one iteration of a home made boosting process:

```python
# Build and fit linear regression model
reg_lm = LinearRegression()
reg_lm.fit(X_train, y_train)

# Calculate the predictions on the test set
pred = reg_lm.predict(X_test)

# Evaluate the performance using the RMSE
rmse = np.sqrt(mean_squared_error(pred,y_test))
print('RMSE: {:.3f}'.format(rmse))

# Now let's do a second iteration:

# Fit a linear regression model to the previous errors
reg_error = LinearRegression()
reg_error.fit(X_train_pop, y_train_error)

# Calculate the predicted errors on the test set
pred_error = reg_error.predict(X_test_pop)

# Evaluate the updated performance
rmse_error = np.sqrt(mean_squared_error(pred_error, y_test_error))
print('RMSE: {:.3f}'.format(rmse_error))
```

This was a somewhat trivial example.  In practice we're more likely to use decision trees for a boosting classifier or regressor.

### Adaptive Boosting (AdaBoost)
This is a classic, and is still widely used, for both classification and regression.  Was proposed by Yoav Freund and Robert Schapire in 97.  Specific features of Adaboost are:

- Instances are drawn using a sample distribution of the training data (like with bagging) but:
		- The distribution starts out uniform
		- Difficult instances are given higher weights
- The estimators are combined with weighte majority voting, higher weights given to more accurate estimators.
- It is guarenteed to get better as the ensemble grows, so long as each estimator has an error rate > 0.5.  However eventually it will over-fit.
- Given the above, there is a trade-off between learning rate and number of estimators.

```python
from sklearn.ensemble import AdaBoostClassifier

clf_ada = AdaBoostClassifier(base_estimator,  # default: decision tree, depth=1
							 n_estimateors=100, # default: 50
							 learning_rate = 1,  # default
							 random_state = 123
							)
```

For regression use `AdaBoostRegressor`,  This has a `loss`  term, by default linear, but could use squre or exponential loss.  Also the decision tree depth by default has a depth of 3.

### Gradient Boosting
Iteratively build up an ensemble model from the sum of the residual errors from the previous estimators.  This is the equivalent of taking the negative of the partial derivative of the square loss with respect to each extimator, hence the concept of *gradient* boosting.
$$Loss = {(F_i(X)-y)^2 \over 2 } $$
Taking the partial derivative derivative of each side:
$$
\begin{aligned}
{\partial Loss \over \partial F_i(X)} &= F_i(X)-y \\
&= - residual
\end{aligned}
$$
```python
from sklearn.ensemble import GradientBoostingClassifier as gbm

clf_gbm = gbm(n_estimators=100, # default
			  learning_rate=0.1 # default
			  max_depth = 3 # default
			  min_samples_split 
			  min_samples_leaf
			  max_features		  
)```

For regression use `GradientBoostingRegressor`, the rest is the same.

#### XGBoost 
- Optimised for GPU computing
- Parallel processing for each estimator.  Not sure how this is actually achieved given the nature of boosting being sequential!  I guess you can still do all the previously created ones in parallel with each iteration?

Usage similar to the others from scikit-learn, though there is no default `learning-rate` or `max_depth`


```python
import xgboost as xgb

clf_xgb = xgb.XGBClassifier(
		n_estimators=100,
		learning_rate=None,
		max_depth=None,
		random_state					
		)

clg_xgb.fit(X_train, y_train)
pred = clf_xgb.predict(X_test)
```

#### LightGBM
Developed by Microsoft in 2017  It is lighter and faster than XGBoost.  Usage is the same as XGboost, except that by default `max_depth=-1`, meaning that it is unlimited.  Consider LightGBM for problems with memory or speed constraints.

#### CatBoost
Open sourced by Yandex in 2017.  Has built in handling of categorical features.  Same advantages as above.  This has been getting more dominant lately in ML competitions, I shoudl give it a try on my Housing Prediction Kaggle benchmark.  Usage is similar, except that no default values at all for the parameters.

## Model Stacking
Split the test data into `train_1` & `train_2`.  With `train_1` train a bunch of different models.  
```python
train_1, train_2 = train_test_split(train_ids, test_size=0.5, random_state=123)
```

With `train_2` use those model outputs as features for the new model.   For example (1) could be a CNN classification with a bunch of different networks.  (2) could use XGBoost or some other tree method to combine the predictions.  That's a pretty neat idea, I'm going to try it some time in a Kaggle comp.

scikit-learn has a built in stacking module.  Usage:
```python
from sklearn.ensemble import StackingClassifier

# A list of tupples of the leel-1 estimators
classifiers = [('clf_1', Classifier1(Params1)), 
			   ('clf_2', Classifier2(Params2))... ]

# Instantiate the second level classifier
clf_meta = ClassifierMeta(ParamsMeta)

clf_stack = StackingClassifier(estimators=classifiers,
							  final_estimator=clf_meta,
							  cv=5 # Number of CV folds
							  stack_method='predict_proba' # Default = auto
							  passthrough=False)
```
`passthrough` is the option to use the original features for the final classifier also.  The stack method could be probabilities, or by default 'auto', which in this case would be class labels.

Similar usage for `StackingRegressor` except that there is no `stack_method`, it is regression by definition.

### The MLxtend Library

This is a third party module, it is lighter and faster but works in a similar syntax to scikit-learn:
```python
from mlxtend.classifier import StackingClassifier
clf_stack = StackingClassifier(classifiers=[clf1, clf2.....]
							   meta_classifier=clf_meta,
							   use_probas=False # default
							   use_features_in_secondary=False # default
								)
```
The only difference I can see is that there is no cross validation, the individual estimators are each trained with the entire feature set.  There is also a regressor option.