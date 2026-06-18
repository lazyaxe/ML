## Analysis of applying Logistic Regression and KNN ML algorithms on MNIST:

## About MNSIT dataset:
The MNIST dataset (Modified National Institute of Standards and Technology dataset) is a large dataset of handwritten digits. Consisting of 70,000 images of handwritten digits.
The images are split in to 784 pixels in 28 * 28 pixel grid each pixel can have a value/brightness from 255(white) to 0 (black).
The records are splits in to two sets, the train set containing 60,000 records of digits and the testing set having 10,000 images.

Here, i have applied Scikit-Learn's KNearestClassifier on two different tasks:
1. Multi-class classification
2. Multi-ouput classification(Image Denoising)
KNN is a non-parametric model so it doesn't have a explict formula.

#### Benchmark Comparision Scikit-Learn's KNearestClassifier vs Scikit-Learn's Logistic Regression:
##### _Logistic Regression_:
Cross Vaildation:

Cross validation accuracy score:  

        Accuracy CV: 0.922
Cross validation F1 weighted score:  

        F1 Score Weighted CV: 0.9217834379064496

Classification Report:
        
                precision    recall  f1-score   support

           0       0.96      0.97      0.96      1356
           1       0.95      0.97      0.96      1611
           2       0.92      0.91      0.91      1401
           3       0.91      0.89      0.90      1443
           4       0.93      0.93      0.93      1336
           5       0.89      0.89      0.89      1316
           6       0.95      0.96      0.95      1402
           7       0.93      0.94      0.93      1455
           8       0.88      0.86      0.87      1333
           9       0.89      0.91      0.90      1347

    accuracy       0.92     14000
    macro avg      0.92      0.92      0.92     14000
    weighted avg   0.92      0.92      0.92     14000

##

Cross Vaildation:
1. Accuracy

        Accuracy CV: 0.9707321428571427
2. F1 Weighted Score

        F1 Score Weighted CV: 0.9707002680070627

Classifiaction report:
        
            precision    recall  f1-score   support

        0       0.97      1.00      0.98      1402
        1       0.95      0.99      0.97      1586
        2       0.98      0.96      0.97      1424
        3       0.97      0.97      0.97      1415
        4       0.98      0.96      0.97      1349
        5       0.96      0.97      0.97      1276
        6       0.98      0.99      0.99      1400
        7       0.96      0.97      0.97      1449
        8       0.99      0.92      0.95      1341
        9       0.96      0.96      0.96      1358

    accuracy                        0.97     14000
    macro avg   0.97      0.97      0.97     14000
    weighted avg0.97      0.97      0.97     14000


#### Challenges faced:
    1. Memory Overflow:
        The memory management was a significant issue as my IDE frequently crashed due to it runnning out memory by storing the dataset in higher order data type float64.
        I had to downcast the data type uint8 as the dataset was eating a lot of memory
    2. High Processing Time:
        Both logistic regression and KNN took a significant amount of time.
        As the logistic regression was doing a lot of matrix operations whereas the KNN had to calculate distances in a very high dimension dataset.
    3. Visualization:
        As this data has 784 axes! Each axis representing each pixel in the image.
        The traditional plots of Precison vs Recall, TPR vs FPR, ROC curve were simply not possible.
        And even some plots(like Accuracy per epoch, Accuracy per K) they were possible but they will be very computationally expensive.


#### Interesting Observations:
1. Less time for fit, predict and cross vaildation for KNN:

        KNN took took the much lesser time to both fit and cross vaildation compared to Logistic Regression
        I think the reason is because of the operations that are done by both algorithms.
        Logistic Regression does matrix mulitplication of a random row with negative error(y_pred - y_true) per number of rows per max iteration in the training phase.
        Meanwhile KNN only stores the dataset as datapoints.

        The KNN only calculates the min-max scaled distance of each datapoint, which IS expensive on high dimensional data but is relatively cheaper than a m * n * max_iter time complexity.
        The slow processing time of Logistic Regression became a pain point in cross validation.

2. Better Performance:
        
        KNN showed a slightly better performance compared to Logistic Regression in this dataset especially in cross vaildation.
3. Min-Max Scaler was better:

        For MNIST dataset, Min-Max Scaler of Scikit-Learn gave is better choice compared to Standard Scaler and harded coded scaling of dividing 
