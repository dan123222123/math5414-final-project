##
using LinearAlgebra
using PaddedViews
using Plots

# verify that A is a stochastic matrix, up to a given tolerance
function check_stochastic(A,tol=10^-4)
  n = size(A,1);
  rerr = sum(A;dims=2) .- ones(n);
  ntrue = (A .>= 0) == trues(n,n)
  return (norm(rerr,2) < tol) && ntrue, rerr, ntrue
end

# scalar splitting operator
function Ŝ(a,λ,r)
  hmin = λ/(a + λ); hmax = a/(a + λ);
  return (a/hmax) * [r (hmax-r); (r-hmin) (1 - r)]
end

# state splitting operator
function ŜM(A,λ,r,k)
  @assert size(A,1) == size(A,2)
  n = size(A,1);
  @assert k <= n

  A₁₁ = A[1:k-1,1:k-1]; c₁ₖ = A[1:k-1,k]; A₁₂ = A[1:k-1,k+1:end];
  rₖ₁ = transpose(A[k,1:k-1]); aₖₖ = A[k,k]; rₖ₂ = transpose(A[k,k+1:end]);
  A₂₁ = A[k+1:end,1:k-1]; c₂ₖ = A[k+1:end,k]; A₂₂ = A[k+1:end,k+1:end];

  ϵ = x -> !isempty(x);

  B1 = hcat(filter(ϵ,[A₁₁,r*c₁ₖ,(1-r)*c₁ₖ,A₁₂])...);
  B2 = hcat(filter(ϵ,[vcat(rₖ₁,rₖ₁),Ŝ(aₖₖ,λ,r),vcat(rₖ₂,rₖ₂)])...)
  B3 = hcat(filter(ϵ,[A₂₁,r*c₂ₖ,(1-r)*c₂ₖ,A₂₂])...);

  return vcat(filter(ϵ,[B1,B2,B3])...)
end
##

##
P = [0.95 0.05;0.45 0.55]

check_stochastic(P)

Q̂ = ŜM(P,0.2,0.7,2)

check_stochastic(Q̂)

Q = [0.95 0.0317 0.0183;0.45 0.475 0.075;0.45 0.275 0.275]

check_stochastic(Q)

eigvals(Q̂) ≈ eigvals(Q)

V = inv(eigvecs(P)); ρP = V[end,:]; ρP = transpose(ρP) / norm(ρP,2);
ρP*P - 0.1*ρP
[0.9 0.1] * P

W = inv(eigvecs(Q̂)); ρQ̂ = W[end,:]; ρQ̂ = transpose(ρQ̂) / norm(ρQ̂,2);
ρQ̂ - 0.1*ρQ̂
[0.9 0.08 0.02] * Q̂

Z = inv(eigvecs(Q)); ρQ = Z[end,:]; ρQ = transpose(ρQ) / norm(ρQ,2);
ρQ - 0.1*ρQ
[0.9 0.07 0.03] * Q
##

##
function ssIEP(λ;visualize=false)
  n = length(λ); λ = sort(λ);
  σ = pop!(λ); A = ones(1,1); # assumed that λ₁ = 1

  if visualize
    plt = heatmap(PaddedView(0,A,(n,n)))
  end

  while !isempty(λ)
    σ = pop!(λ);
    k = argmax(diag(A)); a = diag(A)[k];
    if a <= abs(σ)
      return nothing
    end
    hmin = σ/(a + σ); hmax = a/(a + σ)
    r = rand(range(hmin,hmax,100))
    A = ŜM(A,σ,r,k)
    if visualize
      heatmap!(plt,PaddedView(0,A,(n,n))); display(plt);
    end
  end

  return A
end
##

##
n = 10; λ = rand(n); append!(λ,1); λ = sort(λ); # pre-sort for simpler logic

M = ssIEP(λ;visualize=true)

check_stochastic(M)

norm(eigvals(M) - λ, 2)
##

##
n = 1000; λ = rand(n); append!(λ,1); λ = sort(λ);

M = ssIEP(λ); heatmap(M,yflip=true,clims=(0,10^-4))

check_stochastic(M)

norm(eigvals(M) - λ, 2)
##