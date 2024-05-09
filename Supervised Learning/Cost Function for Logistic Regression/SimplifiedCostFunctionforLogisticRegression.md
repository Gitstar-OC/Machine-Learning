# Simplified Cost Function for Logistic Regression

In the last part you saw the loss function and the cost function for logistic regression. In this part you'll see a slightly simpler way to write out the loss and cost functions, so that the implementation can be a bit simpler when we get to gradient descent for fitting the parameters of a logistic regression model.

Let's take a look. As a reminder, here is the loss function that we had defined in the previous part for logistic regression.
![SCFLR (1)](./../../Assets/Supervised/CFLR/SCFLR%20(1).png)

Because we're still working on a binary classification problem, `y is either zero or one`. Because y is either zero or one and cannot take on any value other than zero or one, we'll be able to come up with a simpler way to write this loss function. You can write the loss function as follows.

![SCFLR (2)](./../../Assets/Supervised/CFLR/SCFLR%20(2).png)

Given a prediction `f of x and the target label y`, the `loss equals negative y times log of f minus 1 minus y times log of 1 minus f`. It turns out this equation, which we just wrote in one line, is completely equivalent to this more complex formula up here. Let's see why this is the case. Remember, y can only take on the values of either one or zero.

In the first case, let's say y equals 1. This first y over here is one and this 1 minus y is 1 minus 1, which is therefore equal to 0. So the loss becomes negative 1 times log of f of x minus 0 times a bunch of stuff.

![SCFLR (4)](./../../Assets/Supervised/CFLR/SCFLR%20(4).png)

That becomes zero and goes away. `When y is equal to 1, the loss is indeed the first term on top, negative log of f of x`.

Let's look at the second case, when y is equal to 0. In this case, this y here is equal to 0, so this first term goes away, and the second term is 1 minus 0 times that logarithmic term. The loss becomes this `negative 1 times log of 1 minus f of x`.

![SCFLR (5)](./../../Assets/Supervised/CFLR/SCFLR%20(5).png)

That's just equal to this second term up here. In the case of y equals 0, we also get back the original loss function as defined above. What you see is that whether y is one or zero, this single expression here is equivalent to the more complex expression up here, which is why this gives us a simpler way to write the loss with just one equation without separating out these two cases, like we did on top. Using this simplified loss function, let's go back and write out the cost function for logistic regression.

Here again is the simplified loss function. Recall that the cost J is just the average loss, average across the entire training set of m examples.

![SCFLR (6)](./../../Assets/Supervised/CFLR/SCFLR%20(6).png)

So it's 1 over m times the sum of the loss from i equals 1 to m. If you plug in the definition for the simplified loss from above, then it looks like this, 1 over m times the sum of this term above. If you bring the negative signs and move them outside, then you end up with this expression over here,
![SCFLR (7)](./../../Assets/Supervised/CFLR/SCFLR%20(7).png)

and this is the cost function. `The cost function that pretty much everyone uses to train logistic regression`. You might be wondering, why do we choose this particular function when there could be tons of other costs functions we could have chosen? Although we won't have time to go into great detail on this in this class, I'd just like to mention that this particular cost function is derived from statistics using a statistical principle called `maximum likelihood estimation`, which is an idea from statistics on how to efficiently find parameters for different models. This cost function has the nice property that it is `convex`.

![SCFLR (8)](./../../Assets/Supervised/CFLR/SCFLR%20(8).png)

But don't worry about learning the details of `maximum likelihood`. It's just a deeper rationale and justification behind this particular cost function.


The upcoming notebook will show you how the logistic cost function is implemented in code. I recommend taking a look at it, because you implement this later into practice lab at the end of the week. This upcoming notebook also shows you how two different choices of the parameters will lead to different cost calculations. You can see in the plot that the better fitting blue decision boundary has a lower cost relative to the magenta decision boundary. So with the simplified cost function, we're now ready to jump into applying gradient descent to logistic regression.
<!--
![SCFLR (3)](./../../Assets/Supervised/CFLR/SCFLR%20(3).png) -->
