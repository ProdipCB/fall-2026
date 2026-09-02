# ECON 6343: Econometrics III
# PS1 :: Question 5
# Prodip Chakraborty


using Test

include("PS1_Prodip_source.jl")


tmpdir = mktempdir()
A,B,C,D = cd(q1,tmpdir)


#------------------------------
# tests for Question 1
#-----------------------------
@testset "q1()" begin

    #--------------------------
    # Question 1, part (a)
    #--------------------------
    @test size(A) == (10,7)
    @test size(B) == (10,7)
    @test size(C) == (5,7)
    @test size(D) == (10,7)
    @test all(A .>= -5)
    @test all(A .<= 10)
    @test C[:,1:5] == A[1:5,1:5]
    @test C[:,6:7] == B[1:5,6:7]
    @test D == A.*(A.<=0)
    @test all(D[A .> 0] .== 0)
    @test all(D[A .<= 0] .== A[A .<= 0])
    A2,B2,C2,D2 = cd(q1,mktempdir())
    @test A == A2
    @test B == B2
    @test C == C2
    @test D == D2

    #--------------------------
    # Question 1, parts (b)
    #--------------------------
    @test length(A) == 70
    #--------------------------
    # Question 1, part (c)
    #--------------------------
    @test length(unique(D)) == sum(A .<= 0) + 1

    #--------------------------
    # Question 1, parts (h) through (k)
    #--------------------------
    @test isfile(joinpath(tmpdir,"matrixpractice.jld"))
    #--------------------------
    # Question 1, parts (i)
    #--------------------------
    @test isfile(joinpath(tmpdir,"firstmatrix.jld"))
    #--------------------------
    # Question 1, parts (j)
    @test isfile(joinpath(tmpdir,"Cmatrix.csv"))
    #--------------------------
    # Question 1, parts (k)
    @test isfile(joinpath(tmpdir,"Dmatrix.dat"))

    #--------------------------
    # Question 1, parts (h)
    #--------------------------
    @test load(joinpath(tmpdir,"firstmatrix.jld"),"A") == A
    @test load(joinpath(tmpdir,"firstmatrix.jld"),"D") == D
    #-------------------------
    # Question 1, parts (i)
    #--------------------------
    @test sort(collect(keys(load(joinpath(tmpdir,"firstmatrix.jld"))))) == ["A","B","C","D"]
    @test sort(collect(keys(load(joinpath(tmpdir,"matrixpractice.jld"))))) == ["A","B","C","D","E","F","G"]

    #--------------------------
    # Question 1, parts (d) through (g)
    #--------------------------
    mp = load(joinpath(tmpdir,"matrixpractice.jld"))
    @test mp["E"] == B[:]
    @test size(mp["F"]) == (2,10,7)
    @test mp["F"][1,:,:] == A
    @test mp["F"][2,:,:] == B
    @test size(mp["G"]) == (50,49)
    @test mp["G"] == kron(B,C)

    #--------------------------
    # Question 1, part (j)
    #--------------------------
    Cin = CSV.read(joinpath(tmpdir,"Cmatrix.csv"),DataFrame)
    @test size(Cin) == (5,7)
    @test Matrix(Cin) ≈ C

    #--------------------------
    # Question 1, part (k)
    #--------------------------
    Din = CSV.read(joinpath(tmpdir,"Dmatrix.dat"),DataFrame,delim='\t')
    @test size(Din) == (10,7)
    @test Matrix(Din) ≈ D
end


#------------------------------
# tests for Question 2
#------------------------------
@testset "q2()" begin

    #--------------------------
    # Question 2, part (f)
    #--------------------------
    @test q2(A,B,C) === nothing

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
    @test AB == AB2
    @test size(AB) == (10,7)

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
    @test Cprime == Cprime2
    @test all(Cprime .>= -5)
    @test all(Cprime .<= 5)
    @test length(Cprime) == sum((C.>=-5) .& (C.<=5))

    #--------------------------
    # Question 2, part (c)
    #--------------------------
    
    N = 500
    K = 6
    T = 5
    X = zeros(N,K,T)
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
    @test size(X) == (N,K,T)
    @test all(X[:,1,:] .== 1.0)
    @test all(in([0.0,1.0]),X[:,2,:])
    @test all(X[:,1,1] .== X[:,1,T])
    @test all(X[:,5,1] .== X[:,5,T])
    @test all(X[:,6,1] .== X[:,6,T])
    @test all(0 .<= X[:,5,1] .<= 20)
    @test all(0 .<= X[:,6,1] .<= 20)
    @test all(X[:,3,1] .≈ 15.0)

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
    @test size(beta) == (K,T)
    @test beta[1,:] == [1.0,1.25,1.5,1.75,2.0]
    @test beta[2,:] ≈ log.(1:T)
    @test beta[3,:] ≈ -sqrt.(1:T)
    @test beta[4,:] ≈ [exp(t)-exp(t+1) for t = 1:T]
    @test beta[5,:] == collect(1.0:T)
    @test beta[6,:] ≈ (1:T)./3

    #--------------------------
    # Question 2, part (e)
    #--------------------------
    Y = hcat([X[:,:,t]*beta[:,t] .+ rand(Normal(0,0.36),N) for t = 1:T]...)
    @test size(Y) == (N,T)
    Y0 = hcat([X[:,:,t]*beta[:,t] for t = 1:T]...)
    for t = 2:T
        @test X[:,:,t]\Y0[:,t] ≈ beta[:,t]
    end

    for t = 1:T
        @test std(Y[:,t] .- X[:,:,t]*beta[:,t]) ≈ 0.36 atol=0.06
    end
end
