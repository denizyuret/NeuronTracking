using FileIO,Optim
include("model.jl")
img = load("raw/509.tiff")
x0 = Float32[290 265 385; 310 360 315] / 512
ygold = convert(Array{Float32}, img.data)
#ygold[ygold.<0.05] = 0.0
x = similar(ygold,(2,3))
x6 = reshape(x, 6)
minit(x)
ypred = similar(ygold)
dx = similar(x)

f(x)=Float32(mlossxy(reshape(x,(2,3)),ygold))

function g(x,dx)
    x33=reshape(x,(2,3))
    dx33=reshape(dx,(2,3))
    mforw(x33,ypred)
    mback(x33,ypred,ygold,dx33)
    dx33
end
# optimize(f, g, x, LBFGS())
