# Heart-disease-prediction
A machine learning Project that implements multiple Algorithms that are trained to detect and predict potential heart diseases. Technology used: Python language including ML Models






ABSTRACT


This project aims to develop a machine learning model to predict the presence of heart disease in patients. The model will be trained on a dataset containing demographic(i.e: age and work type), clinical(i.e: BP and Thallium), and laboratory(i.e: FBS) information of patients. 
The objective is to identify patterns and factors that are correlated with heart disease, enabling doctors to adapt their diagnosis and treatment per patient basis. The project will follow the supervised learning approach, where we will use classification algorithms to predict whether a patient has heart disease or not. We will evaluate the performance of different machine learning algorithms such as logistic regression, decision trees, SVM, KNN and Naive Bayes. 
We will preprocess the data by handling missing values, duplicates, encoding categorical variables, and scaling numerical features. We will then use feature selection models to predict which columns would give us higher accuracy in the models.
Finally, we will evaluate the performance of the model using metrics such as accuracy, precision and recall. 
The results of this project can help medical professionals diagnose heart disease accurately and reduce the fatality caused by cardiovascular diseases.












METHODOLOGY



First we explore the data to see the data types and nulls count.

Dealing with the nulls.

I- First we dealt with the ‘Gender’ column, we calculated the percentages of females and males with and without heart disease.

We had four people with no heart disease, and the percentage of males without heart disease were 45% and females were 54%.
          Therefore, by approximation we filled 2 ‘Male’ and 2 ‘Female’.

The other two people with heart disease were filled with ‘Female’ since the percentage of females with heart disease were 85%.


II- The ‘Age’ column

We drew a boxplot of the age to see if we have any outliers and see the distribution of the data.
Turns out the data is left skewed, therefore we filled the nulls in the age using the median.


III- The ‘Smoking_status’ column
There were four unique values. We filled the nulls with ‘Unknown.


IV- The ‘work_type’ column
There were five unique values, one of the values were ‘never_worked’ , we faced a problem during the SelectKBest feature selection model— will be discussed later in this report — so we replaced that one single value with ‘Self-employed’, like the other nulls in this column.




Dealing with the duplicates.

We call the duplicated() function on the dataset to check for duplicate rows. This returns a boolean mask that indicates which rows are duplicates. We store this mask in the duplicate_rows variable. We then call the drop_duplicates() function on the dataset to remove the duplicate rows. We set the ‘inplace’ parameter to True to modify the original dataset.


Encoding the Data

First we are going to use one-hot encoder on the ‘work_type’ column since it contains four unique values and it won’t really affect the complexity of our model.

We call the fit_transform() method which returns a sparse matrix.

What is a sparse matrix?
It’s a matrix that mostly has zeroes, and it’s used to represent one-hot encoded                        data because it can save memory and computation time.

But we can’t really use a sparse matrix, so we transform it into a dense array using the ‘toarray()’ function.

What is a dense array?
In the context of one-hot encoder, a dense array is the opposite of a sparse matrix.

We transformed the matrix to a dense array so that we can concatenate the produced data with the original dataframe.


Next we are going to encode the rest of the columns using the LabelEncoder class.

For this, we created a for-loop that loops over the data types of each column to see if it’s an object or not. 
If it’s of type: object, the LabelEncoder transforms the categorical data in that column into numerical by giving each unique value a number to represent it.
Though, note that it ranks the categorical data alphabetically, so our model doesn’t accidentally assume that the bigger the number the better.


Normalizing the Data

We used the RobustScaler class in the preprocessing module.
RobustScaler is a feature scaling technique used in machine learning to scale features using statistics that are robust to outliers.

Robustness of machine learning algorithms refers to their ability to perform well and maintain accuracy even when faced with unexpected or noisy data, such as outliers.



Splitting the data into X and Y and training and testing sets using train_test_split.


In the train_test_split() function the default behavior is to split the data randomly. The function shuffles the data before splitting it into training and testing sets, which means that the order of the rows in the original dataframe is not preserved in the resulting training and testing sets. This is done to ensure that the training and testing sets are representative of the overall dataset and to prevent any patterns in the data from affecting the results of the machine learning model.

In the train_test_split() function, the random_state — is a random seed, which is the starting point for the random number generator— parameter is used as the seed for the random number generator that is used to generate the splits. By setting the random_state parameter to a fixed value, you can ensure that the splits generated by the function are reproducible. 
If you don't set the random_state parameter, the function will use a different random seed every time it is called, which can make it difficult to reproduce the same results.
This ensures that the same split is generated every time the function is called with the same input data.





–Quick refresher before we talk about feature selection.
What is correlation?
Correlation is a statistical measure that describes the degree to which two or more variables are related to each other.
It indicates the strength and direction of the relationship between the variables.

Correlation doesn’t imply causation, meaning: just because two variables are correlated, doesn’t necessarily mean that one causes the other.

A correlation coefficient is a numerical measure of the strength and direction of the linear relationship between two variables. It is a statistical concept that helps establish a relationship between predicted and actual values obtained in a statistical experiment.

Feature selection 

We plotted a heat map using the corr() function. 
It computes the correlation coefficient between all columns in the dataframe. The resulting correlation matrix is a square matrix where the diagonal elements are always 1, since they represent the correlation of a variable with itself.
The corr() method can take additional parameters like the method of correlation to be used (e.g, Pearson,Kendall, Spearman)
By default, corr() method calculates pearson’s correlation coefficient through this equation: (sum(xi - mean(x)) * (yi - mean(y))) / (n-1)) / (std(x) * std(y))

The heat map wasn’t a clear representation of the correlation we needed, so we created a bar plot using the corrwith() method.
corrwith() computes the pairwise correlation between the columns of a dataframe and the specific column named ‘Heart Disease’. The resulting coefficients are then plotted as a bar chart using the plot() function.

corr() and corrwith() both compute the pairwise correlation… but they differ in their inputs and outputs.
We can see that ‘Thallium’, ‘number of vessels fluro’, ‘ST depression’, ‘Exercise angina’ and ‘chest pain type’ are highly correlated with the ‘Heart Disease’ column.


L1 regularization

We used Lasso not Ridge regularization because it supports sparsity, meaning it can shrink the coefficients in the loss function to ZERO.

This is achieved by adding an L1 penalty equal to the absolute value of the magnitude of coefficients(the hinge cost/loss function), which limits the size of the coefficients.
The hinge loss function is a specific type of cost function that incorporates a margin or distance from the classification boundary into the cost calculation.
Lasso regularization affects the SVM decision boundary by constraining(shrinking) the size of the estimated coefficients. 
 
The objective function for L1-LASSO is to minimize the sum of square loss, where lambda is the shrinkage factor (penalty) that applies to all the variables.
By minimizing this objective function, LASSO will find a balance using lambda, where a big lambda (big penalty) will force some coefficients to be 0 and leave fewer variables in the model, and vice versa. 

Minimize 1n*i=1n(max(0,1-yixiT))2 +   ||||2


Therefore, Lasso regularization can help to reduce overfitting and improve generalization performance by shrinking some coefficients towards zero and selecting only important features.










SelectKBest

This next feature selection method involves evaluating the linear relationship between each feature and the target. It mainly uses the correlation coefficient to extract the top-k features with the highest correlation coefficient. 
We retained the top 7 features since this was the number of features with a moderate correlation with the target (as identified in the Correlation Plot section of this project). 
Unlike the L1 regularization method, this is a univariate (involving one variable quantity) method, meaning that each feature is considered individually/one-by-one.
The method SelectKBest() takes two parameters, score_func and the number of features to be retained.

The score_func takes two arrays as an input (the column to be compared, and the target column) and returns a single array with the scores.

How are those scores calculated in the scoring function?
The scoring function can be any callable function, in our case, we used the f_classif function.
f_classif computes the F-value for each feature and those are considered the scores which are used to rank the features, and the top-k features are then selected.

What is f-value?
f-value is a statistical measure that represents the ratio of the variance between the group means to the variance within the groups. 


We encountered a runtime warning stating that there were invalid values encountered in the true division, meaning there were cases where the model divided by zero…
In order to overcome this issue, we added a small value called EPSILON, that was a relatively small number, this will add a small constant value EPSILON to the denominator to avoid division by zero or very small numbers.





Random Forest

The final feature selection method considered in this project is the random forest (an ensemble of decision trees). Random forest feature selection works by deriving the importance of each feature in the dataset.
The importance of each feature is calculated based on the decrease in impurity or accuracy when the feature is used in the decision tree.

The *Feature importance score* is then used to rank the features in the dataset and the top-ranked features are selected for the final model.

Implementing this feature selection method involves training a random forest and identifying the features that have importance weight greater than some threshold.
The threshold is set to ‘median’ , so the features with scores above the median will be chosen.
Then the resulting feature set will be used to train the Random Forest Classifier. 

Then we use the get_support() function that returns a boolean mask that determines which features are relevant and we display them.



Now, after we’ve retained a certain number of features from each model, we are going to feed these new feature combinations to an SVM model to see which feature selection model will result in the highest accuracy in the training of our prediction models.

We created a function called eval_svm() where we train an SVM model on the features retained from each feature selection model AND all the features in the dataframe.

The function gives back two parameters, the recall_scorel() and precision_score() scores. 
 recall_score() measures how good the model is at identifying all actual positives out of all positives that exist within a dataset, while precision_score() measures how many of the predicted positives are actually correct out of all positive predictions made by the model.

Based on the average recall, we found that the highest accuracy comes from the features selected by the L1 regularization method, so we split the data once again using those features and trained the models using them.



Logistic Regression

We used GridSearchCV to help us choose the hyperparameters so we can get the best accuracy while avoiding overfitting.

We created an object of the LogisticRegression class and fed it the hyperparametres chen by the grid search, the accuracy of the train and testing set were 83% and 81% simultaneously. 

Before we used grid search, we let the model train on it’s default parameters, if gave us an accuracy of 84% VS  81%
The difference between the fitting of the training and testing data was bigger indicating that there might be a small chance of overfitting.


Decision Tree

Just like Logistic Regression, we used GridSearchCV() class to help us choose the hyperparameters: max_depth and criterion. 

Before the hyperparameter tuning, the Decision tree overfit the training data with a 100% and fit the testing data with a percentage of 69%
After the tuning, the scores of the training and testing improved significantly and became 85% and 72%


Support Vector Machines

We used grid search to tune the parameters. 
Without the hyperparameters the accuracy of the training data was 89% and that of the testing was 83% (overfitting of the training data)
After we performed the grid search, both the training and testing data accuracy became 86% which is optimal.



K-Nearest Neighbor

KNN is a non-parametric supervised machine learning algorithm that is often used for classification, but can also be applied to regression problems. 
It is a distance-based algorithm that classifies objects based on their proximate neighbors' classes. 
The main concept behind KNN is to find the k-nearest neighbors of a point whose class we do not know, and classify it based on the classes of its neighbors.
The  best K-value controls the balance between overfitting and underfitting.

The distance is calculated through euclidean’s distance equation:
d(x, y) = i=1m (Xi - Yi)2


Or through the manhattan/city-block equation:

d(x ,y)= i=1m|Xi -Yi|

To get the best K-value, we created a for loop and let the n_neighbor parameter to i.
We then fit the training and testing data and predict the outcome over a certain number of i.
The error rate for each value of k is being calculated as the mean of the number of incorrect predictions made by the classifier.



Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic machine learning algorithm that can be used in several classification tasks. It is an extension of the Naive Bayes algorithm and is based on applying Bayes' theorem with strong independence assumptions. 
Gaussian Naive Bayes is used for classification tasks with continuous variables and assumes that the data follows a Gaussian or normal distribution.
The algorithm works by calculating the mean and standard deviation values of each input variable for each class value. The mean and standard deviation are used to summarize the distribution of the data. The algorithm then calculates the probabilities for input values for each class using a frequency. With real-valued inputs, we can calculate the mean and standard deviation of input values (x) for each class to summarize the distribution. This means that in addition to the probabilities for each class, we must also store the mean and standard deviations for each input variable for each class.

