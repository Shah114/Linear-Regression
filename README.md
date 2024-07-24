# Linear-Regression
Regression
<br/>

Machine Learning is a branch of Artificial intelligence that focuses on the development of algorithms and statistical models that can learn from and make predictions on data. **Linear regression** is also a type of machine-learning algorithm more specifically a **supervised machine-learning algorithm** that learns from the labelled datasets and maps the data points to the most optimized linear functions. which can be used for prediction on new datasets.<br/>

First of we should know what supervised machine learning algorithms is. It is a type of machine learning where the algorithm learns from labelled data.  Labeled data means the dataset whose respective target value is already known. Supervised learning has two types:<br/>

1. **Classification**: It predicts the class of the dataset based on the independent input variable. Class is the categorical or discrete values. like the image of an animal is a cat or dog?
2. **Regression**: It predicts the continuous output variables based on the independent input variable. like the prediction of house prices based on different parameters like house age, distance from the main road, location, area, etc.<br/>
<br/>

**What is Linear Regression?** <br/>
Linear regression is a type of supervised machine learning algorithm that computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation to observed data.<br/>
When there is only one independent feature, it is known as Simple Linear Regression, and when there are more than one feature, it is known as Multiple Linear Regression. <br/>
Similarly, when there is only one dependent variable, it is considered Univariate Linear Regression, while when there are more than one dependent variables, it is known as Multivariate Regression. <br/>
<br/>

**Why Linear Regression is Important?** <br/>
The interpretability of linear regression is a notable strength. The model’s equation provides clear coefficients that elucidate the impact of each independent variable on the dependent variable, facilitating a deeper understanding of the underlying dynamics. Its simplicity is a virtue, as linear regression is transparent, easy to implement, and serves as a foundational concept for more complex algorithms. <br/>
Linear regression is not merely a predictive tool; it forms the basis for various advanced models. Techniques like regularization and support vector machines draw inspiration from linear regression, expanding its utility. Additionally, linear regression is a cornerstone in assumption testing, enabling researchers to validate key assumptions about the data. <br/>
<br/>

**Types of Linear Regression** <br/>
There are two main types of linear regression:

**Simple Linear Regression**

This is the simplest form of linear regression, and it involves only one independent variable and one dependent variable. The equation for simple linear regression is:

𝑦 =
𝛽`0`
+
𝛽`1`
𝑋


where:

* Y is the dependent variable

* X is the independent variable

* β0 is the intercept

* β1 is the slope

<br/>

**Multiple Linear Regression** <br/>
This involves more than one independent variable and one dependent variable. The equation for multiple linear regression is:

𝑦 =
𝛽
0
+
𝛽
1
𝑋
+
𝛽
2
𝑋
+
…
…
…
𝛽
𝑛
where:

* Y is the dependent variable

* X1, X2, …, Xp are the independent variables

* β0 is the intercept

* β1, β2, …, βn are the slopes <br/>
<br/>

Gradient descent is an optimization algorithm used to minimize the loss function in machine learning and deep learning models. There are several variations of gradient descent, each with its own advantages and drawbacks. Here are the main types:

1. Batch Gradient Descent (BGD) <br/>
Description: In Batch Gradient Descent, the entire dataset is used to compute the gradient of the loss function with respect to the model parameters. <br/>
**Pros:** 
* Converges to the global minimum for convex functions.
* Provides a stable convergence path. <br/>
**Cons:** <br/>
* Requires a lot of memory as it needs to store the entire dataset in memory.
* Can be slow, especially for large datasets. <br/>
2. Stochastic Gradient Descent (SGD) <br/>
Description: Stochastic Gradient Descent updates the model parameters using the gradient of the loss function for each training example individually. <br/>
**Pros:** <br/>
* Faster than Batch Gradient Descent as it updates more frequently.
* Requires less memory since it uses one training example at a time.
* Helps escape local minima due to its stochastic nature. <br/>
**Cons:** <br/>
* Convergence path is noisy, which can lead to oscillations around the minimum.
* May not reach the exact global minimum but rather oscillate around it.
3. Mini-batch Gradient Descent <br/>
Description: Mini-batch Gradient Descent is a compromise between Batch Gradient Descent and Stochastic Gradient Descent. It updates the model parameters using a small batch of training examples. <br/>
**Pros:** <br/>
* Reduces the variance of parameter updates, leading to more stable convergence.
* Allows for vectorization, which can speed up computations.
* Requires less memory than Batch Gradient Descent but more than Stochastic Gradient Descent. <br/>
**Cons:** <br/>
* The choice of batch size can affect the performance and convergence speed.


