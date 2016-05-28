# model parameters
A0 = 0.03
Lambda = 4000f0                 # brightness falls with exp(-0.5*Lambda*r^2)
Tau = 0.001*ones(Float32,4)     # how much x,y,z,a change between timesteps

N = 3                           # number of neurons
X = 512                         # image width
Z = 20                          # number of layers
ZX = 2.0                        # one z layer is ZX pixels in distance
Z0 = 17                         # Z0+1:Z0+Z is one block
Z1 = Z0+div(Z,2)                # middle of the block

# we map 1:X to 0:1 in the xy directions.
X1 = Array(Float32,X,X); for i=1:X,j=1:X; X1[i,j]=i/X; end
X2 = Array(Float32,X,X); for i=1:X,j=1:X; X2[i,j]=j/X; end

# we map 1:Z to 0:ZX*Z/X in the z direction.
zcoor(i)=(mod1(i-Z0,Z) * ZX / X)

# alloc some temporary arrays
M = [ Array(Float32,X,X) for i=1:13 ]
W = [ Array(Float32,4,N) for i=1:7 ]


function zforw{T}(par::Matrix{T},out::Matrix{T}=M[1]; img=Z1,
                  m1::Matrix{T}=M[2], m2::Matrix{T}=M[3])
    fill!(out, A0)                              # A0 is the background brightness
    X3 = zcoor(img)                             # X1,X2,X3 represent pixel positions
    for k=1:size(par,2)
        (x1,x2,x3,a) = par[:,k]                 # x1,x2,x3 represent center position, a is activation
        broadcast!(-, m1, X1, x1)
        broadcast!(*, m1, m1, m1)               # (X1-x1)^2
        broadcast!(-, m2, X2, x2)
        broadcast!(*, m2, m2, m2)               # (X2-x2)^2
        broadcast!(+, m2, m1, m2)               # +(X1-x1)^2
        broadcast!(+, m2, m2, (x3-X3)^2) 	# +(X3-x3)^2
        broadcast!(*, m2, m2, -Lambda/2)
        broadcast!(exp, m2, m2)                 # exp(-Lambda/2 * d2)
        broadcast!(*, m2, m2, a)                # ai * exp(-Lambda/2 * d2)
        broadcast!(+, out, out, m2)             # a0 + sum ai * exp(-Lambda/2 * d2)
    end
    return out
end

# Backward gradient:
function zback{T}(curr::Matrix{T},prev::Matrix{T},pred::Matrix{T},gold::Matrix{T},diff::Matrix{T}=M[4]; img=Z1,
                  delta::Matrix{T}=M[5], dx1::Matrix{T}=M[6], dx2::Matrix{T}=M[7], m1::Matrix{T}=M[8], m2::Matrix{T}=M[9])
    X3 = zcoor(img)
    broadcast!(-, diff, curr, prev)
    broadcast!(*, diff, diff, Tau)
    broadcast!(-, delta, pred, gold)
    for k=1:size(curr,2)
        (x1,x2,x3,a) = curr[:,k]
        broadcast!(-, dx1, X1, x1)
        broadcast!(-, dx2, X2, x2)
        dx3 = X3 - x3
        broadcast!(*, m1, dx1, dx1)
        broadcast!(*, m2, dx2, dx2)
        broadcast!(+, m2, m2, m1)
        broadcast!(+, m2, m2, dx3*dx3)
        broadcast!(*, m2, m2, -Lambda/2)
        broadcast!(exp, m2, m2)
        broadcast!(*, m2, m2, delta) 
        diff[4,k] += mean(m2)                 # dJ/da = sum delta*exp(-Lambda*dist2)
        broadcast!(*, m2, m2, a*Lambda)
        diff[1,k] += mean(broadcast!(*, dx1, dx1, m2))
        diff[2,k] += mean(broadcast!(*, dx2, dx2, m2))
        diff[3,k] += mean(broadcast!(*, m2, m2, dx3))
    end
    return diff
end

function zloss{T}(curr::Matrix{T},prev::Matrix{T},pred::Matrix{T},gold::Matrix{T}; img=Z1,
                  xdiff::Matrix{T}=W[1], ydiff::Matrix{T}=M[10])
    broadcast!(-, xdiff, curr, prev)
    broadcast!(*, xdiff, xdiff, xdiff)
    broadcast!(*, xdiff, xdiff, Tau)
    broadcast!(-, ydiff, pred, gold)
    broadcast!(*, ydiff, ydiff, ydiff)
    return (mean(ydiff) + sum(xdiff))/2
    # return (sum(ydiff) + sum(xdiff))/2
end

# Gradient check:
function zcheck{T}(curr::Matrix{T},prev::Matrix{T},gold::Matrix{T}; img=Z1, eps=T(1e-4),
                   pred::Matrix{T}=M[11], diff::Matrix{T}=W[2])
    zforw(curr, pred, img=img)
    zback(curr, prev, pred, gold, diff, img=img)
    for i=1:length(curr)
        xi = curr[i]
        curr[i] = xi - eps
        zforw(curr, pred, img=img)
        f1 = zloss(curr, prev, pred, gold)
        curr[i] = xi + eps
        zforw(curr, pred, img=img)
        f2 = zloss(curr, prev, pred, gold)
        curr[i] = xi
        df = (f2-f1)/(2*eps)
        println((i,df,diff[i]))
    end
end

# Training:
function ztrain{T}(curr::Matrix{T}, prev::Matrix{T}, gold::Matrix{T}; img=Z1, iter=50, gmin=1f-5, lr=50f0,
                   pred::Matrix{T}=M[12], disp::Matrix{T}=M[13], diff::Matrix{T}=W[3])
    zview(gold)
    for i=1:iter
        zforw(curr, pred, img=img)
        loss = zloss(curr, prev, pred, gold, img=img)
        zback(curr, prev, pred, gold, diff, img=img)
        Base.LinAlg.axpy!(-lr, diff, curr)
        gnorm = vecnorm(diff)
        println((i,:loss,loss,:gnorm,gnorm,:x,round(Int64,X*curr[:])))
        zview(broadcast!(+,disp,gold,pred))
        gnorm < gmin && break
    end
    curr
end

# Tracking
# This version starts from the middle of each block, goes back and forth
function ztrack{T}(y::Array{T}, img1=1, img2=size(y,3);
                   curr::Matrix{T}=W[6], prev::Matrix{T}=W[7])
    while img1 < img2
        i1 = img1; while mod1(i1-Z0,Z) > 1; i1-=1; end
        i3 = img1; while mod1(i3-Z0,Z) < Z; i3+=1; end
        i2 = i1 + 10
        cent = zinit(y[:,:,i2], img=i2)
        copy!(prev, cent); copy!(curr, cent)
        for i=i2:-1:max(1,i1)
            ztrain(curr, prev, y[:,:,i], img=i)
            copy!(prev, curr)
        end
        copy!(prev, cent); copy!(curr, cent)
        for i=i2:i3
            ztrain(curr, prev, y[:,:,i], img=i)
            copy!(prev, curr)
        end
        img1 = i3+1
    end
end

# This loses the centers going from one to the next:
function ztrack1{T}(y::Array{T}, img1=Z1, img2=size(y,3))
    prev = zinit(y[:,:,img1], img=img1)
    curr = copy(prev)
    for i=img1:img2
        ztrain(curr, prev, y[:,:,i], img=i)
        copy!(prev, curr)
    end
end

function zinit{T}(y::Matrix{T};img=Z1)
    x = Array(T,4,N)
    c = zcenters(y)
    for n=1:N
        x[1,n] = c[n][1]/size(y,1)
        x[2,n] = c[n][2]/size(y,2)
        x[3,n] = zcoor(img)
        x[4,n] = c[n][3]-A0
    end
    return x
end

function zcenters{T}(x::Matrix{T}; r=30)
    c = Any[]
    for i=1:size(x,1)
        for j=1:size(x,2)
            zlocalmax(x,i,j,r) && push!(c, (i,j,x[i,j]))
        end
    end
    return sort(c, by=(x->x[3]), rev=true)
end

function zlocalmax{T}(x::Matrix{T}, i, j, r)
    for ii=max(1,i-r):min(size(x,1),i+r)
        for jj=max(1,j-r):min(size(x,2),j+r)
            if x[i,j] < x[ii,jj]
                return false
            end
        end
    end
    return true
end

using ImageView

function zview(y;cmax=maximum(y))
    global _zview
    try
        view(_zview,y')
    catch e
        (_zview,_) = view(y',clim=[0,cmax],name="zview")
    end
end

