# Load _g1 from neuron1.jld which is a 3D z-aligned box holding one neuron

function gloss1(par)            # gaussian, fixed a0
    # Minimum: 2.544195 at [1.0279446 1.04854 0.47514126 1.6086882 1.5215918 3.7772746 0.06005877]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    a0 = 0.03
    (xi,yi,zi,rx,ry,rz,ai)=par
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai * exp(-rx*(xi-x/20)^2 - ry*(yi-y/20)^2 - rz*(zi-z/20)^2)
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss2(par)            # gaussian a0 optimized
    # Minimum: 2.318399 at [1.0074097f0,1.0313389f0,0.4666347f0,4.5696244f0,4.1360598f0,11.593927f0,0.07910681f0,0.04300439f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai,a0)=par
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai * exp(-rx*(xi-x/20)^2 - ry*(yi-y/20)^2 - rz*(zi-z/20)^2)
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss3(par)            # exponential fixed a0
    # Minimum: 2.554641 at [1.0148009f0,1.0381743f0,0.46588328f0,1.3417006f0,1.2868823f0,1.6828519f0,0.09428701f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai)=par
    a0 = 0.03
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai * exp(-rx*abs(xi-x/20) - ry*abs(yi-y/20) - rz*abs(zi-z/20))
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss4(par)            # exponential a0 optimized
    # Minimum: 2.455714 at [1.0124047f0,1.0313879f0,0.4658681f0,2.5158482f0,2.3875716f0,3.5193841f0,0.1388294f0,0.041223526f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai,a0)=par
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai * exp(-rx*abs(xi-x/20) - ry*abs(yi-y/20) - rz*abs(zi-z/20))
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss5(par)            # exp(-r^p) with fixed a0
    # Minimum: 2.525750 at [1.020521f0,1.0451639f0,0.46920383f0,1.4722148f0,1.402376f0,2.522549f0,0.07197181f0,1.4944962f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai,p)=par
    a0 = 0.03
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai * exp(-rx*abs(xi-x/20)^p - ry*abs(yi-y/20)^p - rz*abs(zi-z/20)^p)
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss6(par)            # exp(-r^p) with a0 optimized
    # Minimum: 2.309404 at [1.0071558f0,1.0304579f0,0.46796566f0,6.200063f0,5.4875193f0,20.799503f0,0.06861037f0,0.04346131f0,2.4839554f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai,a0,p)=par
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai * exp(-rx*abs(xi-x/20)^p - ry*abs(yi-y/20)^p - rz*abs(zi-z/20)^p)
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss7(par)            # 1/(1+r^p) with fixed a0
    # Minimum: 2.351727 at [1.0172533f0,1.0372907f0,0.46628773f0,8.777429f0,8.041203f0,20.366737f0,0.11055235f0,1.799353f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai,p)=abs(par)
    a0 = 0.03
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai / (1 + (rx*(xi-x/20)^2 + ry*(yi-y/20)^2 + rz*(zi-z/20)^2)^(p/2))
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss8(par)            # 1/(1+r^p) with optimized a0
    # Minimum: 2.318999 at [1.0090305f0,1.0319712f0,0.4678052f0,5.4503393f0,4.9420156f0,13.582409f0,0.07206944f0,0.04085429f0,3.5884995f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai,a0,p)=abs(par)
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai / (1 + (rx*(xi-x/20)^2 + ry*(yi-y/20)^2 + rz*(zi-z/20)^2)^(p/2))
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss9(par)            # 1/(1+r^2) with fixed a0
    # Minimum: 2.355760 at [1.0174382f0,1.0378766f0,0.46776757f0,6.482576f0,5.9682374f0,15.343796f0,0.09882537f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai)=abs(par)
    a0 = 0.03; p = 2
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai / (1 + (rx*(xi-x/20)^2 + ry*(yi-y/20)^2 + rz*(zi-z/20)^2)^(p/2))
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

function gloss10(par)            # 1/(1+r) with fixed a0
    # Minimum: 2.457183 at [1.0198932f0,1.0372633f0,0.46870422f0,559.0978f0,656.0105f0,1288.5681f0,0.42343268f0]
    global _g1,_g2
    isdefined(:_g1) || error("Define target _g1")
    isdefined(:_g2) || (_g2 = similar(_g1))
    (xi,yi,zi,rx,ry,rz,ai)=abs(par)
    a0 = 0.03; p = 1
    for x=1:41,y=1:41,z=1:20
        _g2[x,y,z] = a0 + ai / (1 + (rx*(xi-x/20)^2 + ry*(yi-y/20)^2 + rz*(zi-z/20)^2)^(p/2))
    end
    vecnorm(broadcast!(-, _g2, _g2, _g1))
end

# optimize(gloss, Float32[1.0,1.0,0.5,1.0,1.0,1.0,.1]; show_trace=true, show_every=10, method=LBFGS())
