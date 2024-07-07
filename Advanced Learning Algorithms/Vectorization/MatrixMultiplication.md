# Matrix Multiplication

You know that a matrix is just a block or 2D array of numbers. What does it mean to multiply two matrices? Let's take a look.

In order to build up to multiplying matrices, let's start by looking at how we take dot products between vectors. Let's use the example of taking the dot product between this vector 1, 2 and this vector 3, 4.

![MM 1](./../../Assets/Algorithms/Vn/MM%20(1).png)

If z is the dot product between these two vectors, then you compute z by multiplying the first element by the first element here, it's 1 times 3, plus the second element times the second element plus 2 times 4, and so that's just 3 plus 8, which is equal to 11.

In the more general case, if z is the dot product between a vector a and vector w, then you compute z by multiplying the first element together and then the second elements together and the third and so on and then adding up all of these products.

![MM 2](./../../Assets/Algorithms/Vn/MM%20(2).png)

That's the vector, vector dot product.

It turns out there's another equivalent way of writing a dot product, which has given a vector a, that is, 1, 2 written as a column. You can turn this into a row. That is, you can turn it from what's called a column vector to a row vector by taking the transpose of a.

![MM 3](./../../Assets/Algorithms/Vn/MM%20(3).png)
The transpose of the vector a means you take this vector and lay its elements on the side like this.

It turns out that if you multiply a transpose, this is a row vector, or you can think of this as a one-by-two matrix with w, which you can now think of as a two-by-one matrix.

![MM 4](./../../Assets/Algorithms/Vn/MM%20(4).png)

Then z equals a transpose times w and this is the same as taking the dot product between a and w.

![MM 5](./../../Assets/Algorithms/Vn/MM%20(5).png)

To recap, z equals the dot product between a and w is the same as z equals a transpose, that is a laid on the side, multiplied by w and this will be useful for understanding matrix multiplication.

![MM 6](./../../Assets/Algorithms/Vn/MM%20(6).png)
That these are just two ways of writing the exact same computation to arrive at z.

Now let's look at vector matrix multiplication, which is when you take a vector and you multiply a vector by a matrix. 

Here again is the vector a 1, 2 and a transpose is a laid on the side, so rather than this think of this as a two-by-one matrix it becomes a one-by-two matrix. Let me now create a two-by-two matrix w with these four elements, 3, 4, 5, 6. 

![MM 7](./../../Assets/Algorithms/Vn/MM%20(7).png)

If you want to compute Z as a transpose times w. Let's see how you go about doing so. 

It turns out that Z is going to be a one-by-two matrix, and to compute the first value of Z we're going to take a transpose, 1, 2 here, and multiply that by the first column of w, that's 3, 4. To compute the first element of Z, you end up with 1 times 3 plus 2 times 4, which we saw earlier is equal to 11, and so the first element of Z is 11. 

![MM 8](./../../Assets/Algorithms/Vn/MM%20(8).png)

Let's figure out what's the second element of Z. It turns out you just repeat this process, but now multiplying a transpose by the second column of w. To do that computation, you have 1 times 5 plus 2 times 6, which is equal to 5 plus 12, which is 17. That's equal to 17. 

![MM 9](./../../Assets/Algorithms/Vn/MM%20(9).png)

Z is equal to this one-by-two matrix, 11 and 17. Now, just one last thing, and then that'll take us to the end of this part, which is how to take vector matrix multiplication and generalize it to matrix matrix multiplication. I have a matrix A with these four elements, the first column is 1, 2 and the second column is negative 1, negative 2 and I want to know how to compute a transpose times w. Unlike the previous slide, A now is a matrix rather than just the vector or the matrix is just a set of different vectors stacked together in columns. 

First let's figure out what is A transpose. In order to compute A transpose, we're going to take the columns of A and similar to what happened when you transpose a vector, we're going to take the columns and lay them on the side, one column at a time. 

![MM 10](./../../Assets/Algorithms/Vn/MM%20(10).png)

The first column 1, 2 becomes the first row 1, 2, let's just laid on side, and this second column, negative 1, negative 2 becomes laid on the side negative 1, negative 2 like this. The way you transpose a matrix is you take the columns and you just lay the columns on the side, one column at a time, you end up with this being A transpose. 

Next we have this matrix W, which going to write as 3,4, 5,6. 

![MM 11](./../../Assets/Algorithms/Vn/MM%20(11).png)

There's a column 3, 4 and the column 5, 6. One way I encourage you to think of matrices. At least there's useful for neural network implementations is if you see a matrix, think of the columns of the matrix and

![MM 12](./../../Assets/Algorithms/Vn/MM%20(12).png)

if you see the transpose of a matrix, think of the rows of that matrix as being grouped together as illustrated here, with A and A transpose as well as W. 

Now, let me show you how to multiply A transpose and W. In order to carry out this computation let me call the columns of A, a_1 and a_2 and that means that a_1 transpose, this the first row of A transpose, and a_2 transpose is the second row of A transpose. 

![MM 13](./../../Assets/Algorithms/Vn/MM%20(13).png)
Then same as before, let me call the columns of W to be w_1 and w_2. 

It turns out that to compute A transpose W, the first thing we need to do is let's just ignore the second row of A and let's just pay attention to the first row of A and let's take this row 1, 2 that is a_1 transpose and multiply that with W. 

![MM 14](./../../Assets/Algorithms/Vn/MM%20(14).png)

You already know how to do that from the previous slide. The first element is 1, 2, inner product or dot product we've 3, 4. That ends up with 3 times 1 plus 2 times 4, which is 11. Then the second element is 1, 2 A transpose, inner product we've 5, 6. There's 5 times 1 plus 6 times 2, which is 5 plus 12, which is 17. That gives you the first row of Z equals A transpose W. All we've done is take a_1 transpose and multiply that by W. That's exactly what we did on the previous slide. 

Next, let's forget a_1 for now, and let's just look at a_2 and take a_2 transpose and multiply that by W. Now we have a_2 transpose times W. To compute that first we take negative 1 and negative 2 and dot product that with 3, 4. That's negative 1 times 3 plus negative 2 times 4 and that turns out to be negative 11. Then we have to compute a_2 transpose times the second column, and has negative 1 times 5 plus negative 2 times 6, and that turns out to be negative 17. 

![MM 15](./../../Assets/Algorithms/Vn/MM%20(15).png)

You end up with A transpose times W is equal to this two-by-two matrix over here. Let's talk about the general form of matrix matrix multiplication. 

This was an example of how you multiply a vector with a matrix, or a matrix with a matrix is a lot of dot products between vectors but ordered in a certain way to construct the elements of the upper Z, one element at a time. 

I know this was a lot, but in the next part, let's look at the general form of how a matrix matrix multiplication is defined and I hope that will make all this clear as well. Let's go on to the next part.
