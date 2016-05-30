# model parameters
A0 = 0.03
Lambda = 2000f0                 # brightness falls with exp(-0.5*Lambda*r^2)
# empirical lambda_x = 1f-6, lambda_a = 1f-2
Tau = Float32[0,0,1f-2,1f-4]  # how much x,y,z,a change between timesteps

N = 3                           # number of neurons
X = 512                         # image width, assuming square X,X images
Z = 20                          # number of layers
ZX = 1.5                        # one z layer is ZX pixels in distance
Z0 = 17                         # Z0+1:Z0+Z is one block
Z1 = Z0+div(Z,2)                # middle of the block
X1drift = 2.5                   # image drift in pixels for first dim
X2drift = 1.0                   # image drift in pixels for second dim

# we map 1:X to 0:1 in the xy directions.
X1 = Array(Float32,X,X); for i=1:X,j=1:X; X1[i,j]=i/X; end
X2 = Array(Float32,X,X); for i=1:X,j=1:X; X2[i,j]=j/X; end

# we map 1:Z to 0:ZX*Z/X in the z direction. some conversions:
# p: pixels, i: images, z: z levels, x: x units
p2x(p::Int)=Float32(p/X)
x2p{T}(x::T)=round(Int,x*X)
z2x(z::Int)=Float32(z*ZX/X)
x2z{T}(x::T)=round(Int,x*X/ZX)
i2z(i::Int)=mod1(i-Z0,Z)
i2x(i::Int)=z2x(i2z(i))

# alloc some temporary arrays
M = [ Array(Float32,X,X) for i=1:20 ]
W = [ Array(Float32,4,N) for i=1:10 ]


function zforw{T}(par::Matrix{T},out::Matrix{T}=M[1]; img=Z1,
                  m1::Matrix{T}=M[2], m2::Matrix{T}=M[3])
    fill!(out, A0)                              # A0 is the background brightness
    X3 = i2x(img)                             # X1,X2,X3 represent pixel positions
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

function zloss{T}(curr::Matrix{T},prev::Matrix{T},pred::Matrix{T},gold::Matrix{T}; img=Z1,
                  xdiff::Matrix{T}=W[1], ydiff::Matrix{T}=M[10])
    broadcast!(-, xdiff, curr, prev)
    broadcast!(*, xdiff, xdiff, xdiff)
    broadcast!(*, xdiff, xdiff, Tau)
    broadcast!(-, ydiff, pred, gold)
    broadcast!(*, ydiff, ydiff, ydiff)
    return (mean(ydiff) + sum(xdiff))/2
end

# Backward gradient:
function zback{T}(curr::Matrix{T},prev::Matrix{T},pred::Matrix{T},gold::Matrix{T},diff::Matrix{T}=M[4]; img=Z1,
                  delta::Matrix{T}=M[5], dx1::Matrix{T}=M[6], dx2::Matrix{T}=M[7], m1::Matrix{T}=M[8], m2::Matrix{T}=M[9])
    X3 = i2x(img)
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
function ztrain{T}(curr::Matrix{T}, prev::Matrix{T}, gold::Matrix{T}; img=Z1, iter=50, gmin=1f-5, gmax=1f-4, lr=0f0,
                   pred::Matrix{T}=M[12], disp::Matrix{T}=M[13], diff::Matrix{T}=W[3], prn=false)
    zdraw(curr, gold, img=img)
    for i=1:iter
        zforw(curr, pred, img=img)
        zback(curr, prev, pred, gold, diff, img=img)
        gnorm = vecnorm(diff)
        gnorm > gmax && scale!(gmax/gnorm, diff)
        lr == 0 && (@show lr = 5f-3/gnorm)
        Base.LinAlg.axpy!(-lr, diff, curr)
        zdraw(curr, gold, img=img)
        vecnorm(diff) < gmin && break
        prn && println((:img,img,:loss,zloss(curr, prev, zforw(curr,pred,img=img), gold, img=img),:gnorm,vecnorm(diff),:x,round(Int64,X*curr[:])))
    end
    loss = zloss(curr, prev, zforw(curr,pred,img=img), gold, img=img)
    println((img,i2z(img),xx2pp(curr)))
    curr
end

function xx2pp{T}(x::Matrix{T})
    a = cell(length(x))
    for i=1:4:length(x)
        a[i] = x2p(x[i])
        a[i+1] = x2p(x[i+1])
        a[i+2] = x2z(x[i+2])
        a[i+3] = round(Int,x[i+3]*0xff)
    end
    return a
end

# This loses the centers going from one block to the next.
# So we need to fix the drift using zfix.
function ztrack{T}(y::Array{T}, i1=Z1, i2=size(y,3); lr=100)
    fixed = zfix(y[:,:,i1], i1)
    prev = zinit(fixed, img=i1)
    curr = copy(prev)
    for i=i1:i2
        fixed = zfix(y[:,:,i], i)
        ztrain(curr, prev, fixed; img=i, lr=lr)
        copy!(prev, curr)
    end
end

function zfix{T}(img::Array{T,2}, i::Int; dx=X1drift, dy=X2drift, out::Array{T,2}=M[15])
    copy!(out,img)
    X,Y = size(img)
    z = i2z(i)
    x = round(Int,dx * z); y = round(Int,dy * z)
    ox1 = max(1,1-x); ox2 = min(X,X-x); oy1 = max(1,1-y); oy2 = min(Y,Y-y)
    ix1 = max(1,1+x); ix2 = min(X,X+x); iy1 = max(1,1+y); iy2 = min(Y,Y+y)
    copy!(sub(out,ox1:ox2,oy1:oy2), sub(img,ix1:ix2,iy1:iy2))
    return out
end

# Tracking
# This version starts from the middle of each block, goes back and forth
function ztrack2{T}(y::Array{T}, img1=1, img2=size(y,3);
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
            i==i2 && copy!(cent, curr)
        end
        copy!(prev, cent); copy!(curr, cent)
        for i=i2+1:i3
            ztrain(curr, prev, y[:,:,i], img=i)
            copy!(prev, curr)
        end
        img1 = i3+1
    end
end

function zinit{T}(y::Matrix{T};img=Z1)
    x = Array(T,4,N)
    c = zcenters(y)
    for n=1:N
        x[1,n] = c[n][1]/size(y,1)
        x[2,n] = c[n][2]/size(y,2)
        x[3,n] = i2x(img)
        x[4,n] = mean(y[c[n][1]-2:c[n][1]+2,c[n][2]-2:c[n][2]+2])-A0
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
        view(_zview,y)
    catch e
        (_zview,_) = view(y,clim=[0,cmax],name="zview",xy=["y","x"])
    end
end

using Images, Colors
using ImageView: annotate!, AnnotationText

function zdraw{T}(curr::Matrix{T}, gold::Matrix{T}; img=Z1, temp::Matrix{T}=M[14])
    global _zdraw
    copy!(temp, gold)
    X3 = i2x(img)                             # X1,X2,X3 represent pixel positions
    # f = a0 + ai * exp(-lambda/2 * d2)
    dmin = -log(0.3)/(Lambda/2)
    dmax = 1f0 # -log(0.05)/(Lambda/2)
    for k=1:size(curr,2)
        (x1,x2,x3,a) = curr[:,k]                 # x1,x2,x3 represent center position, a is activation
        X1a = max(1/X, x1-20/X); X1b = min(1, x1+20/X)
        X2a = max(1/X, x2-20/X); X2b = min(1, x2+20/X)
        for X1=X1a:1/X:X1b, X2=X2a:1/X:X2b
            d2 = (x1-X1)^2 + (x2-X2)^2 + (x3-X3)^2
            dmin <= d2 <= dmax && (temp[x2p(X1),x2p(X2)]=0.0)
        end
    end
    annot = sprint() do s
        println(s, "$img ($(i2z(img)))")
        writedlm(s, reshape(append!(Any["x","y","z","a"], xx2pp(curr)),(4,4)))
    end
    try
        a = first(values(_zdraw[1].annotations))
        a.data.string = annot
        view(_zdraw[1],temp)
    catch e
        _zdraw = view(temp, clim=[0,maximum(temp)], name="zdraw", xy=["y","x"])
        annotate!(_zdraw[1], _zdraw[2], AnnotationText(10, 10, annot, halign="left", valign="top", color=RGB(1,1,1)))
        view(_zdraw[1],temp)
    end
end
