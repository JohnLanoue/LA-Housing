import os
import tarfile
import urllib
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

fetch_housing_data()
df = load_housing_data()
train, test= train_test_split(df, test_size=0.2, random_state=42)

#Places the income category into bins based off median income
df['income_cat']= pd.cut(df['median_income'],bins = [0,1.5,3.0,4.5,6,np.inf],labels=[1,2,3,4,5])



# The below loop iterates on the indexes ensuing that the income_cat is evenly distributed through both test sets.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# From here we can validate the proportions of the column:
strat_test_set['income_cat'].value_counts()/len(strat_test_set)


# Now we know we have the desired test dataframe, we should save this:
housing = strat_train_set.copy()


# ## Visualizing the geography
# If your imagination is ligning up with mine, the below plot looks a lot like a section of California.  The most common areas appear to be the bay area,  LA area and along I5 ~ particulary.



housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# Taking a deeper dive - When incorporating population and house value you will see that the value is in cities and near the beach.  Although, along the beach there still appears to be highly populated areas with a reasonable home value.  The home values along I5 appear to be quite reasonable. Most astonishingly is in the south bay area, you will see some low house value dots engulf high income dotts, a dynamic proves that geography is no the only driver of home value.




housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()


# ## Looking for Correlations
# This is where we begin to really understand the magic that is going into the model.  The closer the values are to 1.000 or -1.000 (EXCEPT median_house_value), the better the correlation is.  As you can see, the only variable that appears to be significant is median_income - what we already have identified as our primary predictor value.  The sort_values function does not do a perfect job of sorting 'winners and loosers', but it makes it easy to fish them out.





housing.drop(['income_cat', 'ocean_proximity'],axis =  1).corr()['median_house_value'].sort_values(ascending=False)


# Below the focus should be on the top row and first column (Not the intersection), where we have the median_house_value - the primary driver.  Again, median income appears to have the strongest correlation, given it cigar shape in both images.  Total rooms also does have a cigar shape to it, but it has minimal

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# Given the significance of median_house_value, it is owed a closer look.  In addition to the aforementioned cigar shape, there appears to be minimal noise in this correlation plot.  However, there does exist a vertical line across the top.  There also appear to be lines going across 450000 and 350000, but they do not appear to span across the whole spectrum of indiviuals making over a median income of 10.
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])


# Because the of the nature of aggregate variable on the significance of the model.  The model may likely preform better if we made some adjustments to these variables, giving them a more accurate representation.
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']

#Need to drop(or encode later) the types.
corr_matrix = housing.drop('ocean_proximity', axis =  1)
corr_matrix = corr_matrix.drop('income_cat',axis =  1).corr()
corr_matrix['median_house_value'].sort_values(ascending = False)


# Preparing the Model
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()


# When treating the nul variables, we are resorting to the median value. Because it is a resiliant variable that will not be affected my major variations in the dataset.
imputer = SimpleImputer(strategy="median")


# The following text will not be subset because median cannot be calculated on ocean_proximity.
housing_num = housing.drop("ocean_proximity", axis=1)


# Now we actually run the imputer on all vlaues.
imputer.fit(housing_num)


# Now we want to look at the the actuall medians produced:
imputer.statistics_


# Now we can apply the imputer to the data.
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing_num.index)


# ## Adjusting Categorical Values

# Mostly for performance reasons, the ocean proximity category would be better represented as digits instead of strings - its all the same to the machine... almost.  The machine may recognize 1<5 even though that is not the case.  Thus we will transform this column into dummy variables.
housing_cat = housing[['ocean_proximity']]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()


# ## Custom transformers
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# ## Transformation Pipelines
# This is a good way for us to impute our planned out transformations in the requested order.
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# # Creating the Model
# ### Linear Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# After building the model and running it, we see that the error is 68,682, this is relatively substantial, particularly on the lower end of houses.

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regression RMSE: ", lin_rmse)


# ### Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# After building the model and running it, we see that the there is no error at all.  Either we have built a perfect model, or there is a good case of overfit.
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Tree RMSE: ", tree_rmse)


# ### Evaluating the Decision Tree
# Now it appears that the estimates are way off and that the tree is overfit to the model. The performance in fact looks worse than the linear model.
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[41]:



print("Tree Model Scores:")
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Linear Model Scores:")
display_scores(lin_rmse_scores)


# ## Random Forest
# This appears to be the best model. The rmse appears to substantially the lowest.
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print("Forest Scores: ")
display_scores(forest_rmse_scores)


# # Fine Tuning
# We will look though the various methods of the hyperparameters to create the best model.

# ## Grid Search
print("Smoke 'em if you got 'em.  We will be running grid searches and they take a while to train on CPU.")
# It appears that the best parameters includes 6 features. And it has given us the 'secret recipe' for building out the model.
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_


# When evaluating all of the geatures, it appears that as mentioned 6/30 was the best option, but also 8/30 seems to be a great option.  For trivia sake, it appears that 2/3 is the worst model combination.

# In[49]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ## Randomized Search
# A more preferred way of looking at the results.  This time, it appears that the features we use is 7/180.  The number of estimators is way higher than any evaluated on the grid, and the features appears be similar to my results above.
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# ## Analyzing the best model
# Below we can tell that most of the fields are important and others can be dropped from the model.
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# # Testing
# At last, we have completed building the model and it is time for us to evaluate the results.
# The final RMSE looks acceptable - not a nobel award winning model though.  The confidence interval does looks relatively narrow and the model is ready for production.

# In[54]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# ## Results
# Present the quantitative results of your experiments, including visualizations
# It appears that there is a substantial difference between the variables that have value and the ones that dont: The leverage of population, total_rooms, households, total_bedrooms, Ocean, Near Ocean, Near Bay and Island are insignificant to the model produced today.
#

# In[56]:


ls = sorted(zip(feature_importances, attributes), reverse=True)
plt.bar([i[1] for i in ls], [i[0] for i in ls])
plt.xticks(rotation = 90)
plt.show()


#
# ## Discussion
# Is your hypothesis supported?
#     Hypothosis 1: One or more of the variables are correlated and make a significant impact on the data:
#     This alternative hypothesis does not stand given that a variable requires a r of .70 to be considered correlated.
#corr_matrix.style.background_gradient(cmap='coolwarm')


# Hypothesis 2:
#     Productive models can be built with the data.  As mentioned above creating a model with the above random forest.  That being said the Results from the the maximized model had a RMSE of 48557, which given the value of the homes, is acceptable.  There still remains much variation and it would be ideal to find better predictors moving forward.
#

# # Conclusion
#
# A successful model has been built and analyzed for determining the value of districts.  Moving forward, we are going to want additional variables for determining this variable.  The primary driver - a known insight is that median_income is the greatest driver for determining the median_house_value.  Currently, it appears that evaluating spikes in income is the greatest means for discovering locations that are a great investment opportunity.
type(sorted(zip(feature_importances, attributes), reverse=True))