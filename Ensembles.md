## Model Blending

Simply take all the models and average their predictions in some sensible way.

For regression, the simplest option would be the mean (or a weighted mean, if you want to give some models more influence than others)

For classification, use a (weighted) geometric mean

If asigning weights to these for some reason, careful not to over-fit.  For example weighting based on CV scores, or public leaderboard scores could do more harm than good.


## Model Stacking

Split the test data into `train_1` & `train_2`.  With `train_1` train a bunch of different models.  

```python
train_1, train_2 = train_test_split(train_ids, test_size=0.5, random_state=123)
```

With `train_2` use those model outputs as features for the new model.   For example (1) could be a CNN classification with a bunch of different networks.  (2) could use XGBoost or some other tree method to combine the predictions.  That's a pretty neat idea, I'm going to try it next comp.