# Unit-13-Homework-Venture-Funding-with-Deep-Learning
![machlearn](https://www.einfochips.com/blog/wp-content/uploads/2018/11/how-to-develop-machine-learning-applications-for-business-featured.jpg)
---


## Assignment
---
In this assignment I was a  risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked me to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.
The business team gave me  a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. With my knowledge of machine learning and neural networks, I decided to use the features in the provided dataset to create a binary classifier model that would predicted whether an applicant will become a successful business. The CSV file contained a variety of information about these businesses, including whether or not they ultimately became successful.

## Instructions
---


1. Prepare the data for use on a neural network model.


2. Compile and evaluate a binary classification model using a neural network.


3. Optimize the neural network model.


## Prepare the data
---

I prepared  the Data for Use on a Neural Network Model
Using my knowledge of Pandas and scikit-learn’s StandardScaler(). I then compiled and evaluated the neural network model .
I opened a starter code file, and completed the following data preparation steps:


1. Read the applicants_data.csv file into a Pandas DataFrame. 
2. Reviewed the DataFrame, looked for categorical variables that  needed to be encoded, as well as columns that could eventually define my features and target variables.


3. I dropped the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they were not relevant to the binary classification model.


4. I encoded the dataset’s categorical variables using OneHotEncoder, and then placed the encoded variables into a new DataFrame.


5. I added the original DataFrame’s numerical variables to the DataFrame containing the encoded variables.







6. I split the features and target sets into training and testing datasets.


7. I used scikit-learn's StandardScaler to scale the features data.
---
## Compile and Evaluate
---


I used my knowledge of TensorFlow to design a binary classification deep neural network model. 

This model  used the dataset’s features to predict whether an Alphabet Soup–funded startup will be successful based on the features in the dataset. 

I considered the number of inputs before determining the number of layers that the model contained or the number of neurons on each layer. 

I then, compiled and fit the model. I evaluated the binary classification model to calculate the model’s loss and accuracy.


I did this by following these  steps:


1. I created a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.




2. I compiled and fit the model using the binary_crossentropy loss function, the adam optimiser, and the accuracy evaluation metric.





3. I evaluated the model using the test data to determine the model’s loss and accuracy.


4. I saved and exported the model to an HDF5 file, and name the file AlphabetSoup.h5.
--- 
## Optimize the neural network model.
---

Optimize the Neural Network Model
Using my knowledge of TensorFlow and Keras, I optimized the model  improved the model's accuracy. 


I completed the following steps:


1. I defined  three new deep neural network models (the original plus 2 optimisation attempts). I improved the first model’s predictive accuracy.

2. I  recalled that accuracy that had a value of 1, so accuracy improves as its value moves closer to 1. To optimized the model for a predictive accuracy as close to 1 as possible I had to do the following


- adjust the input data by dropping different features columns to ensure that no variables or outliers confused the model.


- added more neurons (nodes) to a hidden layer.


- added more hidden layers.


- used different activation functions for the hidden layers.


- added to or reduced the number of epochs in the training regimen.


--- 


I then displayed the accuracy scores achieved by each model, and compared the results.


I then saved the models as an HDF5 file.

![dl](https://t3.ftcdn.net/jpg/00/86/72/58/240_F_86725891_4s8YoGBGizodFi6cpjyvrGRQPmzhIYyD.jpg)