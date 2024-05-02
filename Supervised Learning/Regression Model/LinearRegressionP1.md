# Linear Regression Model Part 1
In this section, we'll look what the overall process of supervised learning is like. Specifically, you see the first model of this course, `Linear Regression Model`. That just means fitting a straight line to your data. It's probably the most widely used learning algorithm in the world today. As you get familiar with linear regression, many of the concepts you see here will also apply to other machine learning models that you'll see later in this specialization. 

Let's start with a problem that you can address using `linear regression`. Say you want to predict the price of a house based on the size of the house. This is the example we've seen earlier. We're going to use a dataset on house sizes and prices from Portland, a city in the United States. Here we have a graph where the horizontal axis is the size of the house in square feet, and the vertical axis is the price of a house in thousands of dollars. 

![LRM1](./../../Assets/Supervised/RegressionModel/LRM1.1.png)

Let's go ahead and plot the data points for various houses in the dataset. Here each data point, each of these little crosses is a house with the size and the price that it most recently was sold for. Now, let's say you're a real estate agent in Portland and you're helping a client to sell her house. She is asking you, how much do you think I can get for this house? This dataset might help you estimate the price she could get for it. You start by measuring the size of the house, and it turns out that the house is 1250 square feet. How much do you think this house could sell for? One thing you could do this, you can build a `linear regression model` from this dataset. 

![LRM2](./../../Assets/Supervised/RegressionModel/LRM1.2.png)

Your model will fit a straight line to the data, which might look like this. Based on this straight line fit to the data, you can see that the house is 1250 square feet, it will intersect the best fit line over here, and if you trace that to the vertical axis on the left, you can see the price is maybe around here, say about $220,000. This is an example of what's called a `supervised learning model`. We call this `supervised learning` because <u> you are first training a model by giving a data that has right answers because you get the model examples of houses with both the size of the house, as well as the price that the model should predict for each house </u>. Well, here are the prices, that is, the right answers are given for every house in the dataset. This `linear regression model` is a particular type of <u> supervised learning model</u>. It's called `regression model` because <u> it predicts numbers as the output like prices in dollars. 


![LRM3](./../../Assets/Supervised/RegressionModel/LRM1.3.png)

Any supervised learning model that predicts a number such as 220,000 or 1.5 or negative 33.2 is addressing what's called a regression problem. Linear regression is one example of a regression model </u>.

But there are other models for addressing regression problems too. In contrast with the regression model, the other most common type of supervised learning model is called a `classification model`. `Classification model predicts categories or discrete categories, such as predicting if a picture is of a cat, meow or a dog, woof, or if given medical record, it has to predict if a patient has a particular disease` .


![LRM4](./../../Assets/Supervised/RegressionModel/LRM1.4.png)

You'll see more about classification models later in this specialization as well. <u> As a reminder about the difference between classification and regression, in classification, there are only a small number of possible outputs. If your model is recognizing cats versus dogs, that's two possible outputs. Or maybe you're trying to recognize any of 10 possible medical conditions in a patient, so there's a discrete, finite set of possible outputs. We call it classification problem, whereas in regression, there are infinitely many possible numbers that the model could output. </u> 


In addition to visualizing this data as a plot here on the left, there's one other way of looking at the data that would be useful, and that's a data table here on the right. 

![LRM5](./../../Assets/Supervised/RegressionModel/LRM1.5.png)

The data comprises a set of inputs. This would be the size of the house, which is this column here. It also has outputs. You're trying to predict the price, which is this column here. 

![LRM6](./../../Assets/Supervised/RegressionModel/LRM1.6.png)

Notice that the horizontal and vertical axes correspond to these two columns, the size and the price. If you have, say, 47 rows in this data table, then there are 47 of these little crosses on the plot of the left, each cross corresponding to one row of the table. For example, the first row of the table is a house with size, 2,104 square feet, so that's around here,

![LRM7](./../../Assets/Supervised/RegressionModel/LRM1.7.png)

and this house is sold for $400,000 which is around here. This first row of the table is plotted as this data point over here. 

![LRM9](./../../Assets/Supervised/RegressionModel/LRM1.9.png)

Now, let's look at some notation for describing the data. This is notation that you find useful throughout your journey in machine learning. 


As you increasingly get familiar with machine learning terminology, this would be terminology they can use to talk about machine learning concepts with others as well since a lot of this is quite standard across AI, you'll be seeing this notation multiple times in this specialization, so it's okay if you don't remember everything for assign through, it will naturally become more familiar overtime. 

![LRM10](./../../Assets/Supervised/RegressionModel/LRM1.10.png)


The dataset that you just saw and that is used to train the model is called a training set. Note that your client's house is not in this dataset because it's not yet sold, so no one knows what the price is. To predict the price of your client's house, you first train your model to learn from the training set and that model can then predict your client's houses price. 

![LRM11](./../../Assets/Supervised/RegressionModel/LRM1.11.png)

`In Machine Learning, the standard notation to denote the input is lowercase x`, and we call this the `input variable`, is also called a **feature** or an **input feature**. For example, for the first house in your training set, x is the size of the house, so x equals 2,104. `The standard notation to denote the output variable which you're trying to predict, which is also sometimes called the target variable, is lowercase y`. Here, y is the price of the house, and for the first training example, this is equal to 400, so y equals 400. 


The dataset has one row for each house and in this training set, there are 47 rows with each row representing a different training example. We're going to use lowercase m to refer it to the total number of training examples, and so here m is equal to 47. 

![LRM12](./../../Assets/Supervised/RegressionModel/LRM1.12.png)

To indicate the single training example, we're going to use the notation parentheses x, y. For the first training example, (x, y), this pair of numbers is (2104, 400). Now we have a lot of different training examples. We have 47 of them in fact. To refer to a specific training example, this will correspond to a specific row in this table on the left, I'm going to use the notation `x superscript in parenthesis, i, y superscript in parentheses i`.

![LRM13](./../../Assets/Supervised/RegressionModel/LRM1.13.png)

 The superscript tells us that this is the ith training example, such as the first, second, or third up to the 47th training example. I here, refers to a specific row in the table. For instance, here is the first example, when i equals 1 in the training set, and so x superscript 1 is equal to 2,104 and y superscript 1 is equal to 400 and let's add this superscript 1 here as well. 
 
![LRM14](./../../Assets/Supervised/RegressionModel/LRM1.14.png)

Just to note, **this superscript i in parentheses is not exponentiation. When I write this, this is not x squared. This is not x to the power 2. It just refers to the second training example.** 

![LRM15](./../../Assets/Supervised/RegressionModel/LRM1.15.png)

 This i, is just an index into the training set and refers to row i in the table. In this part, you saw what a training set is like, as well as a standard notation for describing this training set. In the next part, you will see at what rotate to take this training set that you just saw and feed it to learning algorithm so that the algorithm can learn from this data.

