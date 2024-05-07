# Feature Engineering

The choice of features can have a huge impact on your learning algorithm's performance. In fact, `for many practical applications, choosing or entering the right features is a critical step to making the algorithm work well`. In this part, let's take a look at how you can choose or engineer the most appropriate features for your learning algorithm. Let's take a look at feature engineering by revisiting the example of predicting the price of a house. 

Say you have two features for each house. X_1 is the width of the lot size of the plots of land that the house is built on. This in real state is also called the frontage of the lot, and the second feature, x_2, is the depth of the lot size of, lets assume the rectangular plot of land that the house was built on. 

![FE1](./../../Assets/Supervised/GDP/FE1.png)

Given these two features, x_1 and x_2, you might build a model like this where f of x is w_1x_1 plus w_2x_2 plus b, where x_1 is the frontage or width, and x_2 is the depth. This model might work okay. 

But here's another option for how you might choose a different way to use these features in the model that could be even more effective. You might notice that the area of the land can be calculated as the frontage or width times the depth.

![FE2](./../../Assets/Supervised/GDP/FE2.png)

You may have an intuition that the area of the land is more predictive of the price, than the frontage and depth as separate features. You might define a new feature, `x_3`, `as x_1 times x_2`. This new feature x_3 is equal to the area of the plot of land. 

With this feature, you can then have a model `f_w, b of x` `equals w_1x_1 plus w_2x_2 plus w_3x_3 plus b` 
![FE3](./../../Assets/Supervised/GDP/FE3.png)
so that the model can now choose parameters w_1, w_2, and w_3, depending on whether the data shows that the frontage or the depth or the area x_3 of the lot turns out to be the most important thing for predicting the price of the house.

What we just did, creating a new feature is an example of what's called `feature engineering`, in which you might use your knowledge or intuition about the problem to design new features usually by transforming or combining the original features of the problem in order to make it easier for the learning algorithm to make accurate predictions. 
![FE4](./../../Assets/Supervised/GDP/FE4.png)

Depending on what insights you may have into the application, rather than just taking the features that you happen to have started off with sometimes by defining new features, you might be able to get a much better model. That's feature engineering. It turns out that this one flavor of feature engineering, that allow you to fit not just straight lines, but curves, non-linear functions to your data. Let's take a look in the next part at how you can do that.

