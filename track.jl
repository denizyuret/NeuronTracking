# Full model:
# Data:  f(x,t): pixel intensity at position x, time t
# Model: f(x,t) = a0 + Σi a(i,t) exp(-b(i,t) |x-x(i,t)|^2)
# a(i,t) = N[a(i,t-1), va]
# b(i,t) = N[b(i,t-1), vb]
# x(i,t) = N[x(i,t-1), vx]

include("explore.jl")
include("model.jl")
include("optim.jl")

function track(y)
    
end
