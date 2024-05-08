# Logistic Regression 

Let's talk about logistic regression, which is probably the single most widely used classification algorithm in the world. This is something that I use all the time in my work. 

Let's continue with the example of classifying whether a tumor is malignant. 

![LR1 (2)](./../../Assets/Supervised/CLR/LR1%20(2).png)

Whereas before we're going to use the label 1 or yes to the positive class to represent malignant tumors, and zero or no and negative examples to represent benign tumors. Here's a graph of the dataset where the horizontal axis is the tumor size and the vertical axis takes on only values of 0 and 1, because is a classification problem. 

You saw in the last part that linear regression is not a good algorithm for this problem. In contrast, what logistic regression we end up doing is fit a curve that looks like this, S-shaped curve to this dataset. For this example, if a patient comes in with a tumor of this size, which I'm showing on the x-axis, then the algorithm will output 0.7 suggesting that is closer or maybe more likely to be malignant and benign. 

![LR1 (3)](./../../Assets/Supervised/CLR/LR1%20(3).png)

Will say more later what 0.7 actually means in this context. But the output label y is never 0.7 is only ever 0 or 1. To build out to the logistic regression algorithm, there's an important mathematical function I like to describe which is called the `Sigmoid function`, sometimes also referred to as the logistic function. The `Sigmoid function` looks like this. 
![LR1 (4)](./../../Assets/Supervised/CLR/LR1%20(4).png)

Notice that the x-axis of the graph on the left and right are different. In the graph to the left on the x-axis is the tumor size, so is all positive numbers. Whereas in the graph on the right, you have 0 down here, and the horizontal axis takes on both negative and positive values and have label the horizontal axis Z. I'm showing here just a range of negative 3 to plus 3. `So the Sigmoid function outputs value is between 0 and 1`. If I use g of z to denote this function, then the formula of g of z is equal to 1 over 1 plus e to the negative z.

![LR1 (5)](./../../Assets/Supervised/CLR/LR1%20(5).png)

Where here `e is a mathematical constant that takes on a value of about 2.7`, and so e to the negative z is that mathematical constant to the power of negative z. Notice if z where really be, say a 100, e to the negative z is e to the negative 100 which is a tiny number. So this ends up being 1 over 1 plus a tiny little number, and so the denominator will be basically very close to 1. Which is why when z is large, g of z that is a Sigmoid function of z is going to be very close to 1. 

![LR1 (6)](./../../Assets/Supervised/CLR/LR1%20(6).png)

Conversely, you can also check for yourself that when z is a very large negative number, then g of z becomes 1 over a giant number, which is why g of z is very close to 0. That's why the sigmoid function has this shape where it starts very close to zero and slowly builds up or grows to the value of one. Also, in the Sigmoid function when z is equal to 0, then e to the negative z is e to the negative 0 which is equal to 1, and so g of z is equal to 1 over 1 plus 1 which is 0.5, so that's why it passes the vertical axis at 0.5. 

Now, let's use this to build up to the logistic regression algorithm. We're going to do this in two steps. In the first step, I hope you remember that a straight line function, like a linear regression function can be defined as w. product of x plus b. Let's store this value in a variable which I'm going to call z, and this will turn out to be the same z as the one you saw on the previous slide, but we'll get to that in a minute. The next step then is to take this value of z and pass it to the `Sigmoid function`, also called the `logistic function`, `g`. 

![LR1 (7)](./../../Assets/Supervised/CLR/LR1%20(7).png)

Now, g of z then outputs a value computed by this formula, 1 over 1 plus e to the negative z. There's going to be between 0 and 1. When you take these two equations and put them together, they then give you the logistic regression model f of x, which is equal to g of wx plus b. Or equivalently g of z, which is equal to this formula over here. 

![LR1 (8)](./../../Assets/Supervised/CLR/LR1%20(8).png)

This is the `logistic regression model`, and what it does is it inputs feature or set of features X and outputs a number between 0 and 1. 

Next, let's take a look at how to interpret the output of logistic regression. We'll return to the tumor classification example. The way I encourage you to think of logistic regressions output is to think of it as outputting the probability that the class or the label y will be equal to 1 given a certain input x. For example, in this application, where x is the tumor size and y is either 0 or 1, if you have a patient come in and she has a tumor of a certain size x, and if based on this input x, the model I'll plus 0.7, then what that means is that the model is predicting or the model thinks there's a 70 percent chance that the true label y would be equal to 1 for this patient. 

![LR1 (9)](./../../Assets/Supervised/CLR/LR1%20(9).png)

In other words, the model is telling us that it thinks the patient has a 70 percent chance of the tumor turning out to be malignant. Now, let me ask you a question. See if you can get this right. We know that y has to be either 0 or 1, so if y has a 70 percent chance of being 1, what is the chance that it is 0? So y has got to be either 0 or 1, and thus the probability of it being 0 or 1 these two numbers have to add up to one or to a 100 percent chance. 

That's why if the chance of y being 1 is 0.7 or 70 percent chance, then the chance of it being 0 has got to be 0.3 or 30 percent chance. If someday you read research papers or blog pulls of all logistic regression, sometimes you see this notation that f of x is equal to p of y equals 1 given the input features x and with parameters w and b. 

![LR1 (10)](./../../Assets/Supervised/CLR/LR1%20(10).png)

What the semicolon here is used to denote is just that w and b are parameters that affect this computation of what is the probability of y being equal to 1 given the input feature x? For the purpose of this class, don't worry too much about what this vertical line and what the semicolon mean. You don't need to remember or follow any of this mathematical notation for this class. I'm mentioning this only because you may see this in other places. 

In the notebook that follows this part, you also get to see how the `Sigmoid function` is implemented in code. You can see a plot that uses the `Sigmoid function` so as to do better on the classification tasks that you saw in the previous notebook. Remember that the code will be provided to you, so you just have to run it. I hope you take a look and get familiar with the code. 

Congrats on getting here. You now know what is the logistic regression model as well as the mathematical formula that defines logistic regression. For a long time, a lot of Internet advertising was actually driven by basically a slight variation of logistic regression. This was very lucrative for some large companies, and this is basically the algorithm that decided what ad was shown to you and many others on some large websites. Now, there's, even more, to learn about this algorithm. In the next part, we'll take a look at the details of logistic regression. We'll look at some visualizations and also examines something called the `decision boundary`. This will give you a few different ways to map the numbers that this model outputs, such as 0.3, or 0.7, or 0.65 to a prediction of whether y is actually 0 or 1. Let's go on to the next part to learn more about logistic regression.

