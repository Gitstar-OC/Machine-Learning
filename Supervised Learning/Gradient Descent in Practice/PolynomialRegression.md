# Polynomial Regression

So far we've just been fitting straight lines to our data. Let's take the ideas of multiple linear regression and feature engineering to come up with a new algorithm called `polynomial regression`, `which will let you fit curves, non-linear functions, to your data`.

Let's say you have a housing data-set that looks like this
![FEPR6](./../../Assets/Supervised/GDP/FEPR6.png)

where feature x is the size in square feet. It doesn't look like a straight line fits this data-set very well.

Maybe you want to fit a curve, maybe a quadratic function to the data like this

![FEPR1](./../../Assets/Supervised/GDP/FEPR1.png)

which includes a size x and also x squared, which is the size raised to the power of two. Maybe that will give you a better fit to the data.

But then you may decide that your quadratic model doesn't really make sense because a quadratic function eventually comes back down. Well, we wouldn't really expect housing prices to go down when the size increases. Big houses seem like they should usually cost more.

Then you may choose a cubic function where we now have not only x squared, but x cubed.
![FEPR3](./../../Assets/Supervised/GDP/FEPR3.png)
Maybe this model produces this curve here, which is a somewhat better fit to the data because the size does eventually come back up as the size increases. These are both examples of polynomial regression, because you took your optional feature x, and raised it to the power of two or three or any other power. In the case of the cubic function, the first feature is the size, the second feature is the size squared, and the third feature is the size cubed.

I just want to point out one more thing, which is that if you create features that are these powers like the square of the original features like this, then feature scaling becomes increasingly important. If the size of the house ranges from say, 1-1,000 square feet, then the second feature, which is a size squared, will range from one to a million, and the third feature, which is size cubed, ranges from one to a billion.

![FEPR4](./../../Assets/Supervised/GDP/FEPR4.png)

These two features, x squared and x cubed, take on very different ranges of values compared to the original feature x. `If you're using gradient descent, it's important to apply feature scaling to get your features into comparable ranges of values.

Finally, just one last example of how you really have a wide range of choices of features to use. Another reasonable alternative to taking the size squared and size cubed is to say use the square root of x.

![FEPR5](./../../Assets/Supervised/GDP/FEPR5.png)

Your model may look like w_1 times x plus w_2 times the square root of x plus b. The square root function looks like this, and it becomes a bit less steep as x increases, but it doesn't ever completely flatten out, and it certainly never ever comes back down. This would be another choice of features that might work well for this data-set as well. You may ask yourself, how do I decide what features to use? Later in the notes of `Advanced Algorithm`, you will see how you can choose different features and different models that include or don't include these features, and you have a process for measuring how well these different models perform to help you decide which features to include or not include. 

For now, I just want you to be aware that you have a choice in what features you use. By using feature engineering and polynomial functions, you can potentially get a much better model for your data. 

In the notebook that follows this part, you will see some code that implements polynomial regression using features like x, x squared, and x cubed. Please take a look and run the code and see how it works. There's also another notebook after that one that shows how to use a popular open source toolkit that implements linear regression. `Scikit-learn` is a very widely used open source machine learning library that is used by many practitioners in many of the top AI, internet, machine learning companies in the world.

If either now or in the future you're using machine learning in your job, there's a very good chance you'll be using tools like `Scikit-learn` to train your models. Working through that notebook will give you a chance to not only better understand linear regression, but also see how this can be done in just a few lines of code using a library like `Scikit-learn`. For you to have a solid understanding of these algorithms, and be able to apply them, I do think is important that you know how to implement linear regression yourself and not just call some `scikit-learn` function that is a black-box. But `scikit-learn also has an important role in a way machine learning is done in practice today`. 

Congratulations on finishing this section. Please do take a look at the practice quizzes and also the practice lab, which I hope will let you try out and practice ideas that we've discussed. In this week's practice lab, you implement linear regression. I hope you have a lot of fun getting this learning algorithm to work for yourself. Best of luck with that. In the next section we'll go beyond regression, that is predicting numbers, to talk about our first classification algorithm, which can predict categories.
<!-- 
![FEPR2](./../../Assets/Supervised/GDP/FEPR2.png) -->