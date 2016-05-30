_dx = 2.5
_dy = 1.0
_d = Array(Float32,512,512)

dlimit(a::Int,b::Int,c::Int)=(a<b ? b : a>c ? c : a)

function dview{T}(data::Array{T,3}, img::Int; temp::Matrix{T}=_d, dx=_dx, dy=_dy)
    z = mod1(img-Z0,Z)
    dx = round(Int64,dx * z)
    dy = round(Int64,dy * z)
    X,Y = size(temp)
    @inbounds for x=1:X, y=1:Y
        xx = dlimit(x+dx, 1, X)
        yy = dlimit(y+dy, 1, Y)
        temp[x,y] = data[xx,yy,img]
    end
    zview(temp)
end

