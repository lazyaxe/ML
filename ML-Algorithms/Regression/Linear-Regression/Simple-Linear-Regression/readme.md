### Analysis of implementing Simple Linear Regression in Python using NumPy and Pandas:
Here, i have implemented the simple linear regression function in Python using NumPy and Pandas and tested my implementation on a toy dataset compared it with Scikit-Learn.
#### Benchmark of comparasion of My implementation vs Scikit-Learn's Implementation of Simple Linear Regression:
1. Intercept: `My` = 23437.21046340505 vs `Sklearn` = 23437.21046340505
2. Coefficient: `My` = 9569.586885432866 `Sklearn` = 9569.58688543
3. Mean Absolute Error: `My` = 6802.779572073905 `Sklearn` = 6802.779572073905
4. Mean Squared Error: `My` = 56137509.99782566 `Sklearn` = 56137509.99782566
5. Root Mean Squared Error: `My` = 7492.49691343451 `Sklearn` = 7492.49691343451
6. R2 Score: `My` = 0.8886956733784561 `Sklearn` = 0.8886956733784561

#### _Challenges faced_:<br>
1. Type checking was an issue because the NumPy methods didn't worked with the Pandas DataFrames provided by the user. I solved it by accepting only DataFrames and converting them to NumPy arrays.
2. Another and a bigger challenge i faced is with the compatability of my toy dataset with the Scikit-Learn API methods as only support 2D NumPy arrays and Pandas DataFrames and do not support 1D arrays for the X, independent variable in various solutions.
The workaround of this problem is to reshape the array in (-1, 1) by the reshape function
`array.reshape(-1, 1)`
