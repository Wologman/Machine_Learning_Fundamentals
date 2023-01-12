By splitting out a single holdout set from the test set (or relying on the public test set in a Kaggle comp), the problem is that all our performace improvements are based on the same validation set.  So we end up over-fitting this model rather than generating a true performance metric based on unseen data.

By using cross validation, creating five or ten different folds, with 80% or 90% of the data in each fold, but a different validation set, we greatly reduce this problem.

```python
from sklearn.model_selection import KFold

# Create a KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=123)

# Loop through each cross val split in a dataframe object 'train'
for train_index, test_index in kf.split(train):
	#get the training and test data for teh corresponding split
	cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
```

If there is a lot of class imbalance, or small data size then use stratified K-fold to preserve the target class distribution.  In this case we need to pass the target class variable into the method.

```python
from sklearn.model_selection import StratifiedKFold

# Create a KFold object
str_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Loop through each cross val split in a dataframe object 'train'
for train_index, test_index in str_kf.split(train, train['some_target']):
	#get the training and test data for teh corresponding split
	cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
```

If we produce say two different strategies with models based on 5 folds each, we can compare the models with the standard deviation of error of the 5 folds.  Or compare an overall mean error plus one estandard deviation  (if we are trying to minimise error, subtract if it's a performance metric we want to maximise).

This is computationally very expensive of course, to train on 5 folds will take exactly 5 times longer than training a single fold.

Note, to actually use all five models would require ensembling them, which is computationally expensive come inference time.  Or re-training on all the data once all hyperparameters (including number of epochs of training) and architecture is decided.  This is the simpler option to implement in a real-life scenario, and the only computational penalty is in the training.  

Note that if using tree based models ensembling 5 models is the equivalent of using more trees.

For a real world problem be careful if I'm also doing hyperparameter optimisation at the same time as model performance estimation.  It is important to use CV for hyperparameter tuning, or othewise the likely result is hyperparameters based on choice of validation set, rather than the best hyperparameters.

If I want to both estimate model performance, and do hyperparameter optimisation then this would require a nested cross validation scheme (equivalent to holding out a seperate test set), or the performance will be over-estimated.  For Kaggle, all I need to do is compare models, absolute estimation of performance is unnecessary. So a single CV scheme is adequate.

### Some rules of thumb

- Train on 5 folds, add mean and SD, use that as overall CV.    Do this when exploring different models, or hyperparameters with imbalanced or small datasets to avoid making decisions based on a particular fold choice.
- Retrain on all data at the end, or ensemble 5 models.  Just do this for Kaggle, no need for a real world problem as it complicates the maintenance of the model.  The point of a good model is that the variation between folds should be small anyway, so work on achieving that.
- Use k-fold cross validation on one fold only.  This could be helpful for imbalanced datasets, and is no worse than just a standard train:val:test approach in terms of train dataset size.  So good practice, for example in my predator classification problem.
- No CV, just train:val:test.  OK if dataset is really large & imbalance isn't an issue.
- Nested k-fold for train, val & test.  Do this if working with a small/imbalanced dataset & accurate performance estimate is required as well as model selection.
