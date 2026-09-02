# ECON 6343: Econometrics III
# PS1 :: source code
# Prodip Chakraborty


using JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions


#------------------------------
# question 1
#------------------------------
function q1()

    #--------------------------
    # Question 1, part (a)
    #--------------------------
    
    Random.seed!(1234)
    A = rand(Uniform(-5,10),10,7)
    B = rand(Normal(-2,15),10,7)
    C = [A[1:5,1:5] B[1:5,end-1:end]]
    D = A.*(A.<=0)

    #--------------------------
    # Question 1, part (b)
    #--------------------------
   
    println("number of elements of A is ",length(A))

    #--------------------------
    # Question 1, part (c)
    #--------------------------
  
    println("number of unique elements of D is ",length(unique(D)))

    #--------------------------
    # Question 1, part (d)
    #--------------------------
    
    E = reshape(B,length(B),1)
   
    E = B[:]

    #--------------------------
    # Question 1, part (e)
    #--------------------------

    F = cat(A,B,dims=3)

    #--------------------------
    # Question 1, part (f)
    #--------------------------
   
    F = permutedims(F,(3,1,2))

    #--------------------------
    # Question 1, part (g)
    #--------------------------
  
    G = kron(B,C)
    try
        kron(C,F)
    catch e
        println("kron(C,F) fails with a ",typeof(e),": kron is only defined ",
                "for vectors and matrices, and F is 3-dimensional")
    end

    #--------------------------
    # Question 1, part (h)
    #--------------------------
    save("matrixpractice.jld","A",A,"B",B,"C",C,"D",D,"E",E,"F",F,"G",G)

    #--------------------------
    # Question 1, part (i)
    #--------------------------
    save("firstmatrix.jld","A",A,"B",B,"C",C,"D",D)

    #--------------------------
    # Question 1, part (j)
    #--------------------------
    CSV.write("Cmatrix.csv",DataFrame(C,:auto))

    #--------------------------
    # Question 1, part (k)
    #--------------------------
    CSV.write("Dmatrix.dat",DataFrame(D,:auto),delim='\t')

    #--------------------------
    # Question 1, part (l)
    #--------------------------
    return A,B,C,D
end


#--------------------------------------
# Question 2
#--------------------------------------

function q2(A,B,C)

    #--------------------------
    # Question 2, part (a)
    #--------------------------
  
    AB = zeros(size(A))
    for r in axes(A,1)
        for c in axes(A,2)
            AB[r,c] = A[r,c]*B[r,c]
        end
    end
  
    AB2 = A.*B
    println("AB and AB2 agree: ",isequal(AB,AB2))

    #--------------------------
    # Question 2, part (b)
    #--------------------------
   
    Cprime = Float64[]
    for c in axes(C,2)
        for r in axes(C,1)
            if C[r,c] >= -5 && C[r,c] <= 5
                push!(Cprime,C[r,c])
            end
        end
    end
   
    Cprime2 = C[(C.>=-5) .& (C.<=5)]
    println("Cprime and Cprime2 agree: ",isequal(Cprime,Cprime2))

    #--------------------------
    # Question 2, part (c)
    #--------------------------
    N = 15_169
    K = 6
    T = 5
    X = zeros(N,K,T)

    #clumn 1: intercept
    #clumn 2: dummy variable
    #clumn 3: continuous normal variable
    #clumn 4: normal
    
    for i in axes(X,1)
        X[i,1,:] .= 1.0
        X[i,5,:] .= rand(Binomial(20,0.6))
        X[i,6,:] .= rand(Binomial(20,0.5))
        for t in axes(X,3)
            X[i,2,t] = rand() <= 0.75*(6-t)/5
            X[i,3,t] = rand(Normal(15+t-1,5*(t-1)))
            X[i,4,t] = rand(Normal(pi*(6-t)/3,1/exp(1)))
        end
    end

    #--------------------------
    # Question 2, part (d)
    #--------------------------
  
    beta = zeros(K,T)
    beta[1,:] = [1+0.25*(t-1) for t = 1:T]
    beta[2,:] = [log(t)       for t = 1:T]
    beta[3,:] = [-sqrt(t)     for t = 1:T]
    beta[4,:] = [exp(t)-exp(t+1) for t = 1:T]
    beta[5,:] = [t            for t = 1:T]
    beta[6,:] = [t/3          for t = 1:T]

    #--------------------------
    # Question 2, part (e)
    #--------------------------
   
    Y = hcat([X[:,:,t]*beta[:,t] .+ rand(Normal(0,0.36),N) for t = 1:T]...)
    println("size of Y is ",size(Y))

    #--------------------------
    # Question 2, part (f)
    #--------------------------
    return nothing
end
