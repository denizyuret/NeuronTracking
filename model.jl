# The first attempt did not work.  Try ignoring the background pixels for loss or use cutoff.
# Centers at the edge have a large gradient.
# Small centers have no gradient.
# Few bright pixels close by beat lots of bright pixels far away and capture the center.

# Forward model:
# Input: (3,k) matrix of x,y,a triples.
# Output: (n,n) matrix of brightness values.
# Using picture coordinates.  [0,1]->[0,512]
# Ignore 3D for now.
# x[1,k],x[2,k] in [0,1], x[3,k] in [.2,.4]?

# lambda = 2500.0
lambda = 1000.0
alpha = 0.2
baseline = 0.0

function mforw{T}(x::Matrix{T},y::Matrix{T})
    fill!(y,0)
    _,K = size(x)
    I,J = size(y)
    fill!(y, baseline)
    for k=1:K
        for j=1:J
            for i=1:I
                y[i,j] += alpha * exp(-lambda * ((x[1,k]-i/I)^2 + (x[2,k]-j/J)^2))
            end
        end
    end
    return y
end

# Loss function:

function mloss{T}(ypred::Matrix{T},ygold::Matrix{T})
    0.5*vecnorm(ypred-ygold)^2
end

function mlossxy{T}(x::Matrix{T},ygold::Matrix{T})
    ypred = similar(ygold)
    mforw(x, ypred)
    mloss(ypred, ygold)
end

# Backward gradient:
function mback{T}(x::Matrix{T},ypred::Matrix{T},ygold::Matrix{T},dx::Matrix{T})
    fill!(dx,0)
    _,K = size(x)
    I,J = size(ypred)
    for k=1:K
        x1,x2 = x[:,k]
        for j=1:J
            dj = (j/J-x2)
            for i=1:I
                di = (i/I-x1)
                d2 = di*di + dj*dj
                ydiff = ypred[i,j]-ygold[i,j]
                expd2 = exp(-lambda * d2)
                tmp = ydiff * expd2 * 2 * alpha * lambda
                dx[1,k] += tmp * di
                dx[2,k] += tmp * dj
            end
        end
    end
    dx
end

# Initialize randomly:
# At the edges it shoots out to match dark pixels.
function minit{T}(x::Matrix{T}=Array(Float32,2,3))
    for k=1:length(x)
        x[k] = 0.2*(2*rand()-1)+0.6
    end
    x
end

using Base.LinAlg.axpy!
using ImageView
mview(y;cmax=alpha)=view(y',clim=[0,cmax])
mview(c,y;cmax=alpha)=view(c,y',clim=[0,cmax])

# Training:
function mtrain{T}(x::Matrix{T}, ygold::Matrix{T}; iter=40, lr=0.00025)
    global alpha = maximum(ygold)/2
    global yview = copy(ygold)
    c,_ = mview(yview)
    ypred = similar(ygold)
    dx = similar(x)
    for i=1:iter
        mforw(x, ypred)
        loss = Float32(mloss(ypred, ygold))
        mback(x, ypred, ygold, dx)
        axpy!(-lr, dx, x)
        println((i,loss,vecnorm(dx),round(Int64,512*x[:])))
        axpy!(1, ypred, copy!(yview,ygold))
        mview(c, yview)
    end
end

# Gradient check:
function mcheck{T}(x::Matrix{T}, ygold::Matrix{T}; eps=1e-3)
    ypred = similar(ygold)
    dx = similar(x)
    mforw(x, ypred)
    mback(x, ypred, ygold, dx)
    for i=1:length(x)
        xi = x[i]
        x[i] = xi - eps
        mforw(x, ypred)
        f1 = mloss(ypred, ygold)
        x[i] = xi + eps
        mforw(x, ypred)
        f2 = mloss(ypred, ygold)
        df = (f2-f1)/(2*eps)
        println((i,df,dx[i]))
    end
end

