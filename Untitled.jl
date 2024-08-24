x=[1 2;
   3 5;
   5 6]

function add_num(a,b)
    return a*b
end
r = add_num(3,4)
println(r)

function division(a,b)
    return a%b
end
c=division(6,8)
println(c)

module MyModule

function add_num(x::Int,y::Int)::Int
    return x*y
end
r=add_num(5,6)
println("hello $(r)")

# Define a generic function `area`
function area(shape)
    println("Calculating area of a generic shape")
end

# Method for Circle
struct Circle
    radius::Float64
end

function area(c::Circle)
    return π * c.radius^2
end

# Method for Rectangle
struct Rectangle
    length::Float64
    width::Float64
end

function area(r::Rectangle)
    return r.length * r.width
end

# Usage
c = Circle(5.0)
r = Rectangle(3.0, 4.0)

println("Area of circle: $(area(c))")
println("Area of rectangle: $(area(r))")
ARRAYS MATRICES AND LINEAR ALGEBRA OPERATIONS:
# Creating a 1D array (vector)
vector = [1, 2, 3, 4, 5]

# Creating a 2D array (matrix)
matrix = [1 2 3; 
    4 5 6;
    7 8 9]

# Accessing elements
println("Element at index 2 in vector: ", vector[2])
println("Element at row 2, column 3 in matrix: ", matrix[2, 3])

# Modifying elements
vector[3] = 10
matrix[1, 2] = 0

println("Modified vector: ", vector)
println("Modified matrix: ", matrix)

# Basic operations
println("Sum of all elements in vector: ", sum(vector))
println("Transpose of matrix:\n", transpose(matrix))

using LinearAlgebra

A=[1 2;
   3 4]
B=[5 6;
   7 8]
C=A*B
println("Matrix A:\n",A)
println("Matrix B:\n",B)
println("Matrix A*B:\n",C)

inv_A=inv(A)
println(inv(A))
det_a=det(A)
println(det(A))

using Plots

x=1:10
y=x .^ 2
plot(x,y,label="y = x^2",xlabel="x",ylabel ="y",title="Line Plot")

scatter(x,y,label = "Points")
histogram(randn(100), bins=30,xlabel = "Value", ylabel="Frequency",title="Histogram")

using Makie

# Simple 2D line plot
x = 1:0.1:10
y = sin.(x)
lines(x, y, label = "sin(x)", xlabel = "x", ylabel = "sin(x)", title = "Makie Line Plot")

# 3D plot
x = 1:0.1:10
y = sin.(x)
z = cos.(x)
lines(x, y, z, label = "3D plot", xlabel = "x", ylabel = "sin(x)", zlabel = "cos(x)", title = "3D Line Plot")



