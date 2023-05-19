##
# in the space of n × n real, symmetric matrices, solve the symmetric LiPIEP (LiPIEP2) for a given σ ⊂ ℝ using a Newton-like iteration

using LinearAlgebra
using Distributions
using Plots
using Serialization

scriptdir = @__DIR__; scriptdir = scriptdir * "/";

# PROBLEM -- LiPIEP2
# Given {Aᵢ}ᵐᵢ₌₀ ⊂ 𝒮(n) and {λₖ}ⁿₖ₌₁ ⊂ ℝ, find c⃗ = [ c₁, ⋯ , cₘ ]ᵀ ∈ ℝᵐ such that:
# A(c⃗) := A₀ + c₁A₁ + ⋯  + cₘAₘ satisfies σ(A(c⃗)) ⊂ {λₖ}ⁿₖ₌₁

function Ac(A⃗,c⃗)
  n = size(A⃗[1],1); A = A⃗[1];
  for i = 1:n
    A += c⃗[i] * A⃗[i+1];
  end
  return A
end

function formJb(σ,Q,A⃗)
  n = length(σ);
  J = Matrix{Float64}(undef,(n,n));
  b⃗ = Vector{Float64}(undef,n);
  for i=1:n
    q = Q[:,i]; b⃗[i] = q'*A⃗[1]*q;
    for k=1:n
      J[i,k] = q'*A⃗[k+1]*q;
    end
  end
  return J, b⃗
end

function formY(σ,Q,A)
  n = size(A,1); Y = zeros(n,n);
  for i=1:n
    qᵢ = Q[:,i];
    for j=i+1:n
      qⱼ = Q[:,j];
      Y[i,j] = (qᵢ'*A*qⱼ) / (σ[j] - σ[i]);
    end
  end
  return (Y - Y')/norm(Y - Y',2)
end

# Friedland et al. (1987) Method I
function Newton1(σ,A⃗,c₀;ϵ=10^-5,maxit=100)

  # initialization
  c⃗ᵢ = c₀; Aᵢ = Ac(A⃗,c⃗ᵢ);
  Σᵢ,Qᵢ = eigen(Aᵢ,sortby=(λ) -> λ);

  k = 0;
  while (rᵢ = norm(Σᵢ - σ,2)) > ϵ && k < maxit

    Jᵢ,bᵢ = formJb(σ,Qᵢ,A⃗);
    c⃗ᵢ = Jᵢ \ (σ - bᵢ);
    Aᵢ = Ac(A⃗,c⃗ᵢ);
    Σᵢ,Qᵢ = eigen(Aᵢ,sortby=(λ) -> λ);

    k+=1;
  end

  return c⃗ᵢ,eigvals(Aᵢ),rᵢ

end

# See Chu and Golub (2005) 3.2.4, Friedland et al. (1987) Method III
function Newton3(σ,A⃗,c₀;ϵ=10^-5,maxit=100)

  # initialization
  c⃗ᵢ = c₀; Aᵢ = Ac(A⃗,c⃗ᵢ);
  _,Qᵢ = eigen(Aᵢ,sortby=(λ) -> λ);

  k = 0;
  while (rᵢ = norm(Qᵢ'*Aᵢ*Qᵢ - diagm(σ),2)) > ϵ && k < maxit

    Jᵢ,bᵢ = formJb(σ,Qᵢ,A⃗);
    c⃗ᵢ = Jᵢ \ (σ - bᵢ);
    Aᵢ = Ac(A⃗,c⃗ᵢ);

    Y = formY(σ,Qᵢ,Aᵢ);
    Qᵢ = ( (I + Y/2) \ ( (I - Y/2) * Qᵢ' ) )';

    k+=1;
  end

  return c⃗ᵢ,eigvals(Aᵢ),rᵢ

end

struct IEPP
σ
A⃗
end

IEPPfp = scriptdir * "IEPP";
if isfile(IEPPfp)
  cIEPP = deserialize(IEPPfp)
  σ = cIEPP.σ; A⃗ = cIEPP.A⃗;
else
  n = 2;
  σ = sort(rand(n));
  A⃗ = [(A=rand(n,n); A=A'*A) for _=1:n+1];
  serialize(IEPPfp,IEPP(σ,A⃗));
end
##

##
x = range(-10,10,length=100);
g = collect(Iterators.product(x,x));
Ag = (c) -> Ac(A⃗,c);
plotlyjs()

z = norm.(sort.(diag.(getproperty.(qr.(Ag.(g)),:R))) .- Ref(σ),2);
##

##
plt = surface(x,x,z,size=(600,600),colorbar_title="‖ σ(A(c)) - σ ‖₂")
#savefig(plt,scriptdir * "LiPIEP2sp.png")
##

##
z = cond.(Ag.(g);)
plt = surface(x,x,z,zlims=(1,1000),clims=(1,1000),size=(600,600),colorbar_title="κ(A(c))")
#savefig(plt,scriptdir * "LiPIEP2condsp.png")
##

##
# samples
ϵ = 10^-3; gresvec = [];
Threads.@threads for s=1:100

  local n = 2; A⃗ = [(A=rand(n,n); A=A'*A) for _=1:n+1];

  local resvec = [];

  for _=1:100

    local c₀ = rand(n);
    local ĉ,σ̂,r = Newton3(σ,A⃗,c₀;ϵ,maxit=10);
    append!(resvec,r);

  end

  append!(gresvec,mean(resvec))

end

pyplot(); resbin = 10.0 .^ range(-2,2,length=30);
plt = histogram(gresvec,bins=resbin,normalize=:probability,
label="",xscale=:log10,xlim=extrema(resbin),
xlabel="‖ Qᵢ'*Aᵢ*Qᵢ - diagm(σ) ‖₂",
ylabel="probability")
#savefig(plt,scriptdir * "LiPIEP2samphist.png")
##