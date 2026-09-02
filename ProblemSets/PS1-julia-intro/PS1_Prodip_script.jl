# ECON 6343: Econometrics III
# PS1 :: script
# Prodip Chakraborty


include("PS1_Prodip_source.jl")

#------------------------
# question 1
#------------------------
A,B,C,D = q1()

# writeup for question 1:

# part (b) Matrix A has 70 items in total(10 rows and 7 columns)

# part (c) Matrix D also has 70 items, but only 16 are unique because all positive numbers were changed to zero 

# part (d) The easier way to stack matrix B into a single column is to use the B[:] command 

# part (g) The kron function only works on 1D or 2D matrices, so if we use it on a 3D shape it will cause an error


#------------------------
# question 2
#------------------------
q2(A,B,C)

# writeup for question 2:

# part (a) Doing the math with a loop or the A.*B shortcut gives the same result

# part (b) Because Julia stores data by columns, both methods of organizing the data match perfectly

# part (c) In the first time period, column 3 has no variation and stays fixed at 15

# part (e) Matrix Y has 15,169 rows and 5 columns, and its error size stays the same over time

# part (f) function q2() generate returns nothing 
