# Gradient Descent 
In the last part, you saw visualizations of the cost function j and how you can try different choices of the parameters w and b and see what cost value they get you. It would be nice if we had a more systematic way to find the values of w and b, that results in the smallest possible cost, j of w, b. It turns out there's an algorithm called `gradient descent` that you can use to do that. `Gradient descent is used all over the place in machine learning, not just for linear regression, but for training for example some of the most advanced neural network models, also called deep learning models`. 

Deep learning models are something you will learn about in the second course. Learning these two of gradient descent will set you up with one of the most important building blocks in machine learning. Here's an overview of what we'll do with gradient descent.

<!-- 
 -->
You have the cost function j of w, b right here that you want to minimize. In the example we've seen so far, this is a `cost function for linear regression, but it turns out that gradient descent is an algorithm that you can use to try to minimize any function, not just a cost function for linear regression`.

![GD1](./../../Assets/Supervised/GradientDescent/GD1.png)

Just to make this discussion on gradient descent more general, it turns out that gradient descent applies to more general functions, including other cost functions that work with models that have more than two parameters. 

For instance, if you have a cost function J as a function of w_1, w_2 up to w_n and b, your objective is to minimize j over the parameters w_1 to w_n and b. 

![GD2](./../../Assets/Supervised/GradientDescent/GD2.png)

In other words, you want to pick values for w_1 through w_n and b, that gives you the smallest possible value of j. It turns out that gradient descent is an algorithm that you can apply to try to minimize this cost function j as well. 

What you're going to do is just to start off with some initial guesses for w and b. 

![GD3](./../../Assets/Supervised/GradientDescent/GD3.png)

In linear regression, it won't matter too much what the initial value are, so a common choice is to set them both to 0. For example, you can set w to 0 and b to 0 as the initial guess. With the gradient descent algorithm, what you're going to do is, you'll keep on changing the parameters w and b a bit every time to try to reduce the cost j of w, b until hopefully j settles at or near a minimum.
)

One thing I should note is that for some functions j that may not be a bow shape or a hammock shape, it is possible for there to be more than one possible minimum. 

Let's take a look at an example of a more complex surface plot j to see what gradient is doing. 

![GD4](./../../Assets/Supervised/GradientDescent/GD4.png)

`This function is not a squared error cost function`. `For linear regression with the squared error cost function, you always end up with a bow shape or a hammock shape`. `But this is a type of cost function you might get if you're training a neural network model`. 
<!--  -->

Notice the axes, that is w and b on the bottom axis. For different values of w and b, you get different points on this surface, j of w, b, where the height of the surface at some point is the value of the cost function. 

Now, let's imagine that this surface plot is actually a view of a slightly hilly outdoor park or a golf course where the high points are hills and the low points are valleys like so. I'd like you to imagine if you will, that you are physically standing at this point on the hill.

![GD5](./../../Assets/Supervised/GradientDescent/GD5.png)

If it helps you to relax, imagine that there's lots of really nice green grass and butterflies and flowers is a really nice hill. Your goal is to start up here and get to the bottom of one of these valleys as efficiently as possible. What the gradient descent algorithm does is, you're going to spin around 360 degrees and look around and ask yourself, if I were to take a tiny little baby step in one direction, and I want to go downhill as quickly as possible to or one of these valleys. What direction do I choose to take that baby step?

![GD6](./../../Assets/Supervised/GradientDescent/GD6.png)

Well, if you want to walk down this hill as efficiently as possible, it turns out that if you're standing at this point in the hill and you look around, you will notice that the best direction to take your next step downhill is roughly that direction. `Mathematically, this is the direction of steepest descent`. It means that when you take a tiny baby little step, this takes you downhill faster than a tiny little baby step you could have taken in any other direction. After taking this first step, you're now at this point on the hill over here. 

Now let's repeat the process. Standing at this new point, you're going to again spin around 360 degrees and ask yourself, in what direction will I take the next little baby step in order to move downhill? 

![GD7](./../../Assets/Supervised/GradientDescent/GD7.png)

If you do that and take another step, you end up moving a bit in that direction and you can keep going. From this new point, you can again look around and decide what direction would take you downhill most quickly.

Take another step, another step, and so on, until you find yourself at the bottom of this valley, at this local minimum, right here. 

![GD8](./../../Assets/Supervised/GradientDescent/GD8.png)

What you just did was go through multiple steps of gradient descent. It turns out, gradient descent has an interesting property. Remember that you can choose a starting point at the surface by choosing starting values for the parameters w and b. When you perform gradient descent a moment ago, you had started at this point over here. 

Now, imagine if you try gradient descent again, but this time you choose a different starting point by choosing parameters that place your starting point just a couple of steps to the right over here. 

![GD9](./../../Assets/Supervised/GradientDescent/GD9.png)

If you then repeat the gradient descent process, which means you look around, take a little step in the direction of steepest ascent so you end up here. Then you again look around, take another step, and so on. If you were to run gradient descent this second time, starting just a couple steps in the right of where we did it the first time, then you end up in a totally different valley. 

![GD10](./../../Assets/Supervised/GradientDescent/GD10.png)

This different minimum over here on the right. 

The bottoms of both the first and the second valleys are called local minima. 

![GD11](./../../Assets/Supervised/GradientDescent/GD11.png)

Because if you start going down the first valley, gradient descent won't lead you to the second valley, and the same is true if you started going down the second valley, you stay in that second minimum and not find your way into the first local minimum. This is an interesting property of the gradient descent algorithm, and you see more about this later. In this part, you saw how gradient descent helps you go downhill. In the next part, you'll look at the mathematical expressions that you can implement to make gradient descent work.


