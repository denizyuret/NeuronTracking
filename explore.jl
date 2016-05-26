using ImageMagick,FileIO,JLD,FixedPointNumbers
const imgdir = "/mnt/ai/work/askin/150723_exp3_GCAMP_Images"

# Save all images as a list of images
function img2jld()
    global img = Any[]
    for i=1:7002
        i%100==0 && print(".")
        push!(img, load("$imgdir/$i.tiff"))
    end
    println(); info("Saving...")
    JLD.save("img.jld","img",img)
end

# Save all images as a 512x512x7002 array of UFixed16 values
function img3jld(img)
    global img3=Array(UFixed16,512,512,7002)
    for i=1:7002; copy!(img3,1+(i-1)*512*512,img[i].data,1,512*512); end
    JLD.save("img3.jld","img3",img3)
end

# Find centers as local maxima of pixel values: doesn't work very well
function centers(x::Matrix{UFixed16}; r=100)
    c = Any[]
    for i=1:size(x,1)
        for j=1:size(x,2)
            localmax(x,i,j,r) && push!(c, (i,j,x[i,j]))
        end
    end
    return c
end

function localmax(x::Matrix{UFixed16}, i, j, r)
    for ii=max(1,i-r):min(size(x,1),i+r)
        for jj=max(1,j-r):min(size(x,2),j+r)
            if x[i,j] < x[ii,jj]
                return false
            end
        end
    end
    return true
end

# Find the image with the max pixel for every 20 image block
function findmax20()
    zz = Int[]
    for i=0:20:size(img3,3)
        d = img3[:,:,i+1:min(i+20,7002)]
        (v,n) = findmax(d)
        (x,y,z) = ind2sub(d,n)
        push!(zz,z)
    end
    zz
end

# Find variation in activation looking at the value of the max pixel for every 20 image block
function findmaxpix()
    zz = Float32[]
    for i=0:20:size(img3,3)
        d = img3[:,:,i+1:min(i+20,7002)]
        push!(zz,maximum(d))
    end
    zz
end

# Find variation based on distance around the max pixel in every block
function pixeldistance(; r=50)
    dr = Float32[]; dv = Float32[]
    for i=0:20:size(img3,3)
        d = img3[:,:,i+1:min(i+20,7002)]
        (v,n) = findmax(d)
        (v > 0.5) && continue
        (x,y,z) = ind2sub(d,n)
        for xx = max(1,x-r):min(512,x+r)
            for yy = max(1,y-r):min(512,y+r)
                vv = d[xx,yy,z]
                push!(dr, sqrt((x-xx)^2 + (y-yy)^2))
                push!(dv, Float32(vv)-Float32(v))
            end
        end
    end
    (dr, dv)
end

function brighten(img)
    c = copy(img)
    for i=1:length(c.data)
        c.data[i] = min(1, 5*c.data[i])
        # c.data[i] = max(0, 1-4*c.data[i])
    end
    return c
end
