module FiniteDifferences

function central_difference(f,x::Vector{T},h::T) where T<:Real
    f0 = f(x)
    N = length(x)
    f_dim = size(f0)
    idx = [Colon() for i=1:length(f_dim)] # used to index things
    A = zeros(T,(f_dim...,N)) # f can have arbitrary dimensions here
    Δ1 = zeros(T,N)
    Δ2 = zeros(T,N)
    Δ1[:] = x[:]
    Δ2[:] = x[:]
    for i=1:N
        Δ1[i] = x[i]-h
        Δ2[i] = x[i]+h
        if i > 1
            Δ1[i-1] = x[i-1]
            Δ2[i-1] = x[i-1]
        end
        diff =  f(Δ2) - f(Δ1)
        A[vcat(idx,i)...] = diff/2h
    end
    return A
end

end