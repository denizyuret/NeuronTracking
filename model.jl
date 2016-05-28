# The first attempt did not work.  Try ignoring the background pixels for loss or use cutoff.
# Centers at the edge have a large gradient.
# Small centers have no gradient.
# Few bright pixels close by beat lots of bright pixels far away and capture the center.
# We need to initialize well and use the model to correct for small shifts.

# Forward model:
# Input: (3,k) matrix of x,y,a triples.
# Output: (n,n) matrix of brightness values.
# Using picture coordinates.  [0,1]->[0,512]
# Ignore 3D for now.
# x[1,k],x[2,k] in [0,1], x[3,k] in [.2,.4]?

lambda = 2000.0
baseline = 0.032
coor1 = Array(Float32,512,512); for i=1:512,j=1:512; coor1[i,j]=i/512; end
coor2 = Array(Float32,512,512); for i=1:512,j=1:512; coor2[i,j]=j/512; end
temp = [ Array(Float32,512,512) for i=1:10 ]

function mforw{T}(par::Matrix{T},img::Matrix{T})
    fill!(img, baseline)
    for k=1:size(par,2)
        (x,y,a) = par[:,k]
        broadcast!(-, temp[1], coor1, x)
        broadcast!(*, temp[1], temp[1], temp[1])
        broadcast!(-, temp[2], coor2, y)
        broadcast!(*, temp[2], temp[2], temp[2])
        broadcast!(+, temp[2], temp[1], temp[2])
        broadcast!(*, temp[2], temp[2], -lambda)
        broadcast!(exp, temp[2], temp[2])
        broadcast!(*, temp[2], temp[2], a)
        broadcast!(+, img, img, temp[2])
    end
    return img
end

function mforw0{T}(par::Matrix{T},img::Matrix{T})
    _,K = size(par)
    I,J = size(img)
    fill!(img, baseline)
    @inbounds @simd for j=1:J
        @inbounds for i=1:I
            @inbounds for k=1:K
                (x,y,a)=par[:,k]
                img[i,j] += a * exp(-lambda * ((x-i/I)^2 + (y-j/J)^2))
            end
        end
    end
    return img
end

# Backward gradient:
function mback{T}(par::Matrix{T},pred::Matrix{T},gold::Matrix{T},diff::Matrix{T})
    fill!(diff,0)
    broadcast!(-, temp[10], pred, gold)
    for k=1:size(par,2)
        (x,y,a) = par[:,k]
        broadcast!(-, temp[1], coor1, x)
        broadcast!(-, temp[2], coor2, y)
        broadcast!(*, temp[3], temp[1], temp[1])
        broadcast!(*, temp[4], temp[2], temp[2])
        broadcast!(+, temp[5], temp[3], temp[4])
        broadcast!(*, temp[5], temp[5], -lambda)
        broadcast!(exp, temp[5], temp[5])
        broadcast!(*, temp[5], temp[5], temp[10]) # delta*exp(-lambda*d2)
        broadcast!(*, temp[6], temp[5], 2*a*lambda)
        broadcast!(*, temp[1], temp[1], temp[6])
        broadcast!(*, temp[2], temp[2], temp[6])
        diff[1,k] += sum(temp[1])
        diff[2,k] += sum(temp[2])
        diff[3,k] += sum(temp[5])
    end
    return diff
end

function mback0{T}(par::Matrix{T},pred::Matrix{T},gold::Matrix{T},diff::Matrix{T})
    fill!(diff,0)
    _,K = size(par)
    I,J = size(pred)
    for j=1:J
        for i=1:I
            delta = pred[i,j]-gold[i,j]
            for k=1:K
                (x,y,a) = par[:,k]
                di = (i/I - x)
                dj = (j/J - y)
                d2 = di*di + dj*dj
                tmp1 = delta * exp(-lambda * d2)
                tmp2 = 2 * a * lambda * tmp1
                diff[1,k] += tmp2 * di
                diff[2,k] += tmp2 * dj
                diff[3,k] += tmp1
            end
        end
    end
    diff
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

# Initialize randomly:
# At the edges it shoots out to match dark pixels.
# function minit{T}(x::Matrix{T}=Array(Float32,2,3))
#     for k=1:length(x)
#         x[k] = 0.2*(2*rand()-1)+0.6
#     end
#     x
# end

function minit{T}(x::Matrix{T},y::Matrix{T})
    c = centers(y)
    for k=1:size(x,2)
        x[1,k] = c[k][1]/size(y,1)
        x[2,k] = c[k][2]/size(y,2)
        x[3,k] = c[k][3]-baseline
        # x[4,k] = lambda
    end
    x
end

using Base.LinAlg.axpy!
using ImageView

function mview(y;cmax=maximum(y))
    global _mview
    try
        view(_mview,y')
    catch e
        (_mview,_) = view(y',clim=[0,cmax],name="mview")
    end
end

# mview(y;cmax=alpha)=view(y',clim=[0,cmax])
# mview(c,y;cmax=alpha)=view(c,y',clim=[0,cmax])

# Training:
function mtrain{T}(x::Matrix{T}, ygold::Matrix{T}; iter=50, gmin=1.0, lr=0.0004)
    global yview = copy(ygold)
    alpha = maximum(ygold)/2
    mview(yview)
    ypred = similar(ygold)
    dx = similar(x)
    for i=1:iter
        mforw(x, ypred)
        loss = Float32(mloss(ypred, ygold))
        mback(x, ypred, ygold, dx)
        axpy!(-lr, dx, x)
        nx = vecnorm(dx)
        println((i,loss,nx,round(Int64,512*x[:])))
        axpy!(1, ypred, copy!(yview,ygold))
        mview(yview)
        nx < gmin && break
    end
    x
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

