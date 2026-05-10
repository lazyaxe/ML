### Analysis of implementing Multiple Linear Regression in Python using NumPy and Pandas:
Here, i have implemented the multiple linear regression function in Python using NumPy and Pandas and tested my implementation on a toy dataset of diabetes compared it with Scikit-Learn.

### $\hat{Y} = X {\beta}$

#### Where,
#### ${\beta} = ({X}^T {X})^{-1} {X}^T Y $

${\beta}$ = The vector of weight/parameters/features size $m$ x 1 <br>
$m$  = number of features/columns

#### Benchmark of comparasion of My implementation vs Scikit-Learn's Implementation of Simple Linear Regression:
`My Implementation parameters:`
The intercept_ = 153.6223761918323
weight_matrix = [5.194799    -231.20116081   521.84289082   287.84684961
 -1055.54216705   653.95643184   202.3941596    186.41397793
   830.46292542   117.34110977]

`Sklearn Implementation parameters:`
The intercept_ = 153.6223761918321
 weight_matrix = [    5.194799    -231.20116081   521.84289082   287.84684961
 -1055.54216705   653.95643184   202.3941596    186.41397793
   830.46292542   117.34110977]

`My model mean_absolute_error` 47.95454893561524
`Sklearn model mean_absolute_error` 47.954548935615215

`My model mean_squared_error` 3289.5990922612045
`Sklearn model mean_squared_error `3289.5990922612004

`My model root_mean_squared_error` 57.355026739259785
`Sklearn model root_mean_squared_error` 57.35502673925975

#### _Challenges faced_:<br>
1. Type checking was an issue because the NumPy methods didn't worked with the Pandas DataFrames provided by the user. I solved it by accepting only DataFrames and converting them to NumPy arrays.
2. The 1D array issue is not much of an issue here, unless the user specifically provides a 1D array.
