# Coefficients of Dropped Categorical Variables

So, you've dropped some columns after one-hot encoding.  What if you need the value of those columns' coefficients from a linear regression?

### The Dataset

Info provided when I [downloaded it](https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html) was:

Thunder Basin Antelope Study

The data (X1, X2, X3, X4) are for each year.

 - X1 = spring fawn count/100
 - X2 = size of adult antelope population/100
 - X3 = annual precipitation (inches)
 - X4 = winter severity index (1=mild, 5=severe)


```python
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
```


```python
antelope_df = pd.read_csv("antelope_study.csv")
```

The export format gave us an extra empty row and empty column, drop them


```python
antelope_df = antelope_df.drop("Unnamed: 4", axis=1).drop(8)
```

Set the column names to something human-readable


```python
antelope_df.columns = [
    "spring_fawn_count", 
    "adult_antelope_population", 
    "annual_precipitation",
    "winter_severity_index"
]
```

Artificially turning a numeric variable to a categorical variable (two different versions), for example purposes


```python
antelope_df["low_precipitation"] = [int(x < 12) for x in antelope_df["annual_precipitation"]]
antelope_df["high_precipitation"] = [int(x >= 12) for x in antelope_df["annual_precipitation"]]
antelope_df = antelope_df.drop("annual_precipitation", axis=1)
```

One-hot encode the "winter severity index"


```python
ohe = OneHotEncoder(categories=[[1,2,3,4,5]])
antelope_df["winter_severity_index"] = antelope_df["winter_severity_index"].apply(int).astype('category')
winter_severity_ohe_array = ohe.fit_transform(antelope_df[["winter_severity_index"]]).toarray()
winter_severity_df = pd.DataFrame(winter_severity_ohe_array,columns=ohe.categories_[0])

antelope_df = antelope_df.join(winter_severity_df)
antelope_df = antelope_df.drop("winter_severity_index", axis=1)
```

So the final "full" dataframe looks like:


```python
antelope_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spring_fawn_count</th>
      <th>adult_antelope_population</th>
      <th>low_precipitation</th>
      <th>high_precipitation</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.9</td>
      <td>9.2</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4</td>
      <td>8.7</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>7.2</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.3</td>
      <td>8.5</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.2</td>
      <td>9.6</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.9</td>
      <td>6.8</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.4</td>
      <td>9.7</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.1</td>
      <td>7.9</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



(Yes it's quite small and maybe I should redo this with a bigger dataset)

There would be issues if I passed all of the current features into a linear regression, since `low_precipitation` perfectly explains everything in `high_precipitation` (and vice versa), plus each of the "winter severity index" columns is perfectly explained by the other four.  But all of the following models will only be build with a subset of this full dataframe

For all of the following models, the target variable is `spring_fawn_count`


```python
y = antelope_df["spring_fawn_count"]
```

## Model 1

The first model is just taking in the numeric `adult_antelope_population` input in order to predict `spring_fawn_count`


```python
X1 = antelope_df[["adult_antelope_population"]]
X1 = sm.add_constant(X1)

model1 = sm.OLS(y, X1)
results1 = model1.fit()
print(results1.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.881
    Model:                            OLS   Adj. R-squared:                  0.862
    Method:                 Least Squares   F-statistic:                     44.56
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):           0.000547
    Time:                        18:43:29   Log-Likelihood:                 2.2043
    No. Observations:                   8   AIC:                           -0.4086
    Df Residuals:                       6   BIC:                           -0.2498
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -1.6791      0.634     -2.648      0.038      -3.231      -0.127
    adult_antelope_population     0.4975      0.075      6.676      0.001       0.315       0.680
    ==============================================================================
    Omnibus:                        1.796   Durbin-Watson:                   2.235
    Prob(Omnibus):                  0.407   Jarque-Bera (JB):                0.749
    Skew:                          -0.178   Prob(JB):                        0.688
    Kurtosis:                       1.544   Cond. No.                         72.9
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    //anaconda3/envs/mod2-project-env/lib/python3.8/site-packages/numpy/core/fromnumeric.py:2495: FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
      return ptp(axis=axis, out=out, **kwargs)
    //anaconda3/envs/mod2-project-env/lib/python3.8/site-packages/scipy/stats/stats.py:1449: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=8
      warnings.warn("kurtosistest only valid for n>=20 ... continuing "


We could interpret this model as saying, _each additional 100 adult antelopes is associated with an increase of 50 (rounded from 49.75) spring fawns_

## Model 2

Now let's add the `low_precipitation` feature


```python
X2 = antelope_df[[
    "adult_antelope_population",
    "low_precipitation"
]]
X2 = sm.add_constant(X2)

model2 = sm.OLS(y, X2)
results2 = model2.fit()
print(results2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.888
    Model:                            OLS   Adj. R-squared:                  0.844
    Method:                 Least Squares   F-statistic:                     19.88
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):            0.00417
    Time:                        18:43:29   Log-Likelihood:                 2.4459
    No. Observations:                   8   AIC:                             1.108
    Df Residuals:                       5   BIC:                             1.346
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -1.1163      1.213     -0.920      0.400      -4.235       2.003
    adult_antelope_population     0.4396      0.131      3.366      0.020       0.104       0.775
    low_precipitation            -0.1466      0.263     -0.558      0.601      -0.822       0.529
    ==============================================================================
    Omnibus:                        0.411   Durbin-Watson:                   2.269
    Prob(Omnibus):                  0.814   Jarque-Bera (JB):                0.459
    Skew:                          -0.338   Prob(JB):                        0.795
    Kurtosis:                       2.042   Cond. No.                         133.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Comparing Model 1 and Model 2

We have added another feature, which has impacted all of the coefficients

 - `const` went from -1.6791 to -1.1163
 - `adult_antelope_population` went from 0.4975 to 0.4396

Our r-squared improved, so it seems like these coefficients changed "for the better", becoming a better representation of the "true" coefficients.

We could interpret this model as saying _each additional 100 antelopes is associated with an increase of 44 (rounded from 43.96) spring fawns, and low precipitation is associated with a decrease of 15 (rounded from 14.66) spring fawns_

## Model 3

And then let's do the opposite, adding the `high_precipitation` feature instead


```python
X3 = antelope_df[[
    "adult_antelope_population",
    "high_precipitation"
]]
X3 = sm.add_constant(X3)

model3 = sm.OLS(y, X3)
results3= model3.fit()
print(results3.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.888
    Model:                            OLS   Adj. R-squared:                  0.844
    Method:                 Least Squares   F-statistic:                     19.88
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):            0.00417
    Time:                        18:43:30   Log-Likelihood:                 2.4459
    No. Observations:                   8   AIC:                             1.108
    Df Residuals:                       5   BIC:                             1.346
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -1.2629      1.005     -1.256      0.265      -3.847       1.322
    adult_antelope_population     0.4396      0.131      3.366      0.020       0.104       0.775
    high_precipitation            0.1466      0.263      0.558      0.601      -0.529       0.822
    ==============================================================================
    Omnibus:                        0.411   Durbin-Watson:                   2.269
    Prob(Omnibus):                  0.814   Jarque-Bera (JB):                0.459
    Skew:                          -0.338   Prob(JB):                        0.795
    Kurtosis:                       2.042   Cond. No.                         111.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Comparing Model 2 and Model 3

Here we substituted one feature for another feature, which theoretically should provide identical information (the 0s have just become 1s)

Coefficient updates:

 - `const` went from -1.1163 to -1.2629
 - `adult_antelope_population` stayed the same at 0.4396
 - `high_precipitation` is now 0.1466 (whereas its inverse `low_precipitation` was -0.1466)
 
So, what can this tell us about the difference when you drop one vs. the other?

1. Either way, the `adult_antelope_population` coefficient does not change
2. The `const` coefficient does change, because the meaning of "all the features are zero" has changed, but I would argue that this was never a particularly interpretable thing to start with
3. By inverting the 1s and 0s, the precipitation coefficient is pointing in the opposite direction

We could interpret this model as saying _each additional 100 antelopes is associated with an increase of 44 (rounded from 43.96) spring fawns, and high precipitation is associated with a increase of 15 (rounded from 14.66) spring fawns_

### So, what does this mean about a single dropped categorical variable?

First, the intercept changes, because the meaning of "all the feature are zero" has changed.  If one or the other seems more like a true "null hypothesis", it might make more sense for it to be the dropped one.

Second, the coefficient flips depending on the framing.  So, if you used the feature `high_precipitation` in the model, and someone wants to know "what is the impact of low precipitation?", you can give the coefficient for `high_precipitation` * -1

## Model 4

Now, instead of adding a feature to `adult_antelope_population` that can be represented by 1 column (2 categories), let's add one that needs 4 columns (5 categories).


```python
antelope_df.columns
```




    Index([        'spring_fawn_count', 'adult_antelope_population',
                   'low_precipitation',        'high_precipitation',
                                     1,                           2,
                                     3,                           4,
                                     5],
          dtype='object')




```python
X4 = antelope_df[[
    "adult_antelope_population",
    # drop 1
    2, 3, 4, 5
]]
X4 = sm.add_constant(X4)

model4 = sm.OLS(y, X4)
results4= model4.fit()
print(results4.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.982
    Model:                            OLS   Adj. R-squared:                  0.938
    Method:                 Least Squares   F-statistic:                     22.15
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):             0.0438
    Time:                        18:43:30   Log-Likelihood:                 9.8065
    No. Observations:                   8   AIC:                            -7.613
    Df Residuals:                       2   BIC:                            -7.136
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -3.2132      1.069     -3.006      0.095      -7.812       1.386
    adult_antelope_population     0.6818      0.109      6.243      0.025       0.212       1.152
    2                            -0.2205      0.197     -1.118      0.380      -1.069       0.628
    3                            -0.1743      0.195     -0.893      0.466      -1.014       0.665
    4                             0.3044      0.339      0.898      0.464      -1.154       1.763
    5                             0.4771      0.375      1.272      0.331      -1.137       2.091
    ==============================================================================
    Omnibus:                        1.784   Durbin-Watson:                   2.219
    Prob(Omnibus):                  0.410   Jarque-Bera (JB):                0.577
    Skew:                          -0.649   Prob(JB):                        0.749
    Kurtosis:                       2.785   Cond. No.                         201.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


So, the following statements would be reasonable interpretations:

_A weather index of 2 is associated with 22 fewer spring fawns, compared to a weather index of 1_

_A weather index of 3 is associated with 17 fewer spring fawns, compared to a weather index of 1_

_A weather index of 4 is associated with 30 more spring fawns, compared to a weather index of 1_

_A weather index of 5 is associated with 48 more spring fawns, compared to a weather index of 1_

But what if someone asks "what is the impact of weather index 1?"

## Model 5, 6, 7, 8

Let's drop each of the other columns in turn


```python
X5 = antelope_df[[
    "adult_antelope_population",
    # drop 2
    1, 3, 4, 5
]]
X5 = sm.add_constant(X5)

model5 = sm.OLS(y, X5)
results5= model5.fit()
print(results5.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.982
    Model:                            OLS   Adj. R-squared:                  0.938
    Method:                 Least Squares   F-statistic:                     22.15
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):             0.0438
    Time:                        18:43:30   Log-Likelihood:                 9.8065
    No. Observations:                   8   AIC:                            -7.613
    Df Residuals:                       2   BIC:                            -7.136
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -3.4337      0.972     -3.534      0.072      -7.615       0.747
    adult_antelope_population     0.6818      0.109      6.243      0.025       0.212       1.152
    1                             0.2205      0.197      1.118      0.380      -0.628       1.069
    3                             0.0462      0.130      0.355      0.757      -0.514       0.607
    4                             0.5249      0.250      2.096      0.171      -0.553       1.603
    5                             0.6976      0.284      2.461      0.133      -0.522       1.918
    ==============================================================================
    Omnibus:                        1.784   Durbin-Watson:                   2.219
    Prob(Omnibus):                  0.410   Jarque-Bera (JB):                0.577
    Skew:                          -0.649   Prob(JB):                        0.749
    Kurtosis:                       2.785   Cond. No.                         176.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
X6 = antelope_df[[
    "adult_antelope_population",
    # drop 3
    1, 2, 4, 5
]]
X6 = sm.add_constant(X6)

model6 = sm.OLS(y, X6)
results6= model6.fit()
print(results6.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.982
    Model:                            OLS   Adj. R-squared:                  0.938
    Method:                 Least Squares   F-statistic:                     22.15
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):             0.0438
    Time:                        18:43:30   Log-Likelihood:                 9.8065
    No. Observations:                   8   AIC:                            -7.613
    Df Residuals:                       2   BIC:                            -7.136
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -3.3875      0.957     -3.539      0.071      -7.506       0.731
    adult_antelope_population     0.6818      0.109      6.243      0.025       0.212       1.152
    1                             0.1743      0.195      0.893      0.466      -0.665       1.014
    2                            -0.0462      0.130     -0.355      0.757      -0.607       0.514
    4                             0.4787      0.234      2.042      0.178      -0.530       1.487
    5                             0.6514      0.267      2.436      0.135      -0.499       1.802
    ==============================================================================
    Omnibus:                        1.784   Durbin-Watson:                   2.219
    Prob(Omnibus):                  0.410   Jarque-Bera (JB):                0.577
    Skew:                          -0.649   Prob(JB):                        0.749
    Kurtosis:                       2.785   Cond. No.                         172.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
X7 = antelope_df[[
    "adult_antelope_population",
    # drop 4
    1, 2, 3, 5
]]
X7 = sm.add_constant(X7)

model7 = sm.OLS(y, X7)
results7= model7.fit()
print(results7.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.982
    Model:                            OLS   Adj. R-squared:                  0.938
    Method:                 Least Squares   F-statistic:                     22.15
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):             0.0438
    Time:                        18:43:30   Log-Likelihood:                 9.8065
    No. Observations:                   8   AIC:                            -7.613
    Df Residuals:                       2   BIC:                            -7.136
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -2.9088      0.799     -3.640      0.068      -6.347       0.529
    adult_antelope_population     0.6818      0.109      6.243      0.025       0.212       1.152
    1                            -0.3044      0.339     -0.898      0.464      -1.763       1.154
    2                            -0.5249      0.250     -2.096      0.171      -1.603       0.553
    3                            -0.4787      0.234     -2.042      0.178      -1.487       0.530
    5                             0.1727      0.206      0.840      0.489      -0.712       1.057
    ==============================================================================
    Omnibus:                        1.784   Durbin-Watson:                   2.219
    Prob(Omnibus):                  0.410   Jarque-Bera (JB):                0.577
    Skew:                          -0.649   Prob(JB):                        0.749
    Kurtosis:                       2.785   Cond. No.                         150.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
X8 = antelope_df[[
    "adult_antelope_population",
    # drop 5
    1, 2, 3, 4
]]
X8 = sm.add_constant(X8)

model8 = sm.OLS(y, X8)
results8= model8.fit()
print(results8.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.982
    Model:                            OLS   Adj. R-squared:                  0.938
    Method:                 Least Squares   F-statistic:                     22.15
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):             0.0438
    Time:                        18:43:30   Log-Likelihood:                 9.8065
    No. Observations:                   8   AIC:                            -7.613
    Df Residuals:                       2   BIC:                            -7.136
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -2.7361      0.756     -3.619      0.069      -5.989       0.517
    adult_antelope_population     0.6818      0.109      6.243      0.025       0.212       1.152
    1                            -0.4771      0.375     -1.272      0.331      -2.091       1.137
    2                            -0.6976      0.284     -2.461      0.133      -1.918       0.522
    3                            -0.6514      0.267     -2.436      0.135      -1.802       0.499
    4                            -0.1727      0.206     -0.840      0.489      -1.057       0.712
    ==============================================================================
    Omnibus:                        1.784   Durbin-Watson:                   2.219
    Prob(Omnibus):                  0.410   Jarque-Bera (JB):                0.577
    Skew:                          -0.649   Prob(JB):                        0.749
    Kurtosis:                       2.785   Cond. No.                         149.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


## Comparing 4, 5, 6, 7, 8

To summarize:

 - In all cases, the `adult_antelope_population` coefficient did not change, which is consistent with the idea that the model has the same information, no matter which column was dropped
 - The `const` coefficients changed each time
 - The relative coefficients looked like this:
 
| Baseline | 1       | 2       | 3       | 4       | 5      |
| -------- | ------- | ------- | ------- | ------- | ------ |
| 1        |         | -0.2205 | -0.1743 |  0.3044 | 0.4771 |
| 2        |  0.2205 |         |  0.0462 |  0.5249 | 0.6976 |
| 3        |  0.1743 | -0.0462 |         |  0.4787 | 0.6514 |
| 4        | -0.3044 | -0.5249 | -0.4787 |         | 0.1727 |
| 5        | -0.4771 | -0.6976 | -0.1727 | -0.1727 |        |

It's a more complicated pattern, but still fundamentally the same as when there were only two categories

Previously, we could just flip the sign of the coefficient as we flipped the baseline, but now there are as many possible baselines as there are categories

So if someone asks "what is the impact of weather index 1?", you need to ask "against which baseline?"

 - If 2 is the new baseline, flip the coefficient of index 2 with baseline 1, i.e. 0.2205 * (-1) = -0.2205
 - If 3 is the new baseline, flip the coefficient of index 3 with baseline 1, i.e. 0.1743 * (-1) = -0.1743
 - If 4 is the new baseline, flip the coefficient of index 4 with baseline 1, i.e. -0.3044 * (-1) = 0.3044
 - If 5 is the new baseline, flip the coefficient of index 5 with baseline 1, i.e. -0.4771 * (-1) = 0.4771

## A Final Note on Multiple Dropped Categorical Variable Features

A question came up, _does it really make sense to have more than one dropped categorical feature at once?_

Well, whichever category you choose to drop is encoded as a baseline that the other categories diverge from.  So, your model might be more interpretable if you intentionally choose which categories to drop rather than just dropping the first one, but any combination of categories can be a baseline as far as the model is concerned.

Here's a model where the baseline is "not low precipitation and weather index 5":


```python
X9 = antelope_df[[
    "adult_antelope_population",
    "low_precipitation",
    # drop 5
    1, 2, 3, 4
]]
X9 = sm.add_constant(X9)

model9 = sm.OLS(y, X9)
results9 = model9.fit()
print(results9.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.986
    Model:                            OLS   Adj. R-squared:                  0.901
    Method:                 Least Squares   F-statistic:                     11.59
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):              0.221
    Time:                        18:43:30   Log-Likelihood:                 10.702
    No. Observations:                   8   AIC:                            -7.405
    Df Residuals:                       1   BIC:                            -6.849
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -1.8810      1.956     -0.962      0.512     -26.736      22.974
    adult_antelope_population     0.5841      0.239      2.444      0.247      -2.452       3.620
    low_precipitation            -0.1907      0.381     -0.501      0.704      -5.027       4.645
    1                            -0.3845      0.509     -0.755      0.588      -6.852       6.083
    2                            -0.6881      0.359     -1.917      0.306      -5.250       3.874
    3                            -0.5261      0.421     -1.251      0.429      -5.869       4.817
    4                            -0.1336      0.271     -0.492      0.709      -3.582       3.315
    ==============================================================================
    Omnibus:                        0.094   Durbin-Watson:                   1.996
    Prob(Omnibus):                  0.954   Jarque-Bera (JB):                0.310
    Skew:                          -0.000   Prob(JB):                        0.856
    Kurtosis:                       2.035   Cond. No.                         279.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


And here's a model where the baseline is "not high precipitation and weather index 5":


```python
X10 = antelope_df[[
    "adult_antelope_population",
    "high_precipitation",
    # drop 5
    1, 2, 3, 4
]]
X10 = sm.add_constant(X10)

model10 = sm.OLS(y, X10)
results10 = model10.fit()
print(results10.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:      spring_fawn_count   R-squared:                       0.986
    Model:                            OLS   Adj. R-squared:                  0.901
    Method:                 Least Squares   F-statistic:                     11.59
    Date:                Wed, 04 Dec 2019   Prob (F-statistic):              0.221
    Time:                        18:43:30   Log-Likelihood:                 10.702
    No. Observations:                   8   AIC:                            -7.405
    Df Residuals:                       1   BIC:                            -6.849
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    const                        -2.0717      1.635     -1.267      0.425     -22.842      18.699
    adult_antelope_population     0.5841      0.239      2.444      0.247      -2.452       3.620
    high_precipitation            0.1907      0.381      0.501      0.704      -4.645       5.027
    1                            -0.3845      0.509     -0.755      0.588      -6.852       6.083
    2                            -0.6881      0.359     -1.917      0.306      -5.250       3.874
    3                            -0.5261      0.421     -1.251      0.429      -5.869       4.817
    4                            -0.1336      0.271     -0.492      0.709      -3.582       3.315
    ==============================================================================
    Omnibus:                        0.094   Durbin-Watson:                   1.996
    Prob(Omnibus):                  0.954   Jarque-Bera (JB):                0.310
    Skew:                           0.000   Prob(JB):                        0.856
    Kurtosis:                       2.035   Cond. No.                         239.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


The `const` coefficient has changed, and the the `high_precipitation` coefficient has flipped from the sign of the `low_precipitation` coefficient, but the coefficients of the weather indices have not changed, because you did not modify the baseline with respect to them, you only modified the baseline with respect to the precipitation feature.


```python

```
