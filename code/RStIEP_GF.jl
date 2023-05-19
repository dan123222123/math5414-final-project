##
using LinearAlgebra
using DifferentialEquations
using Distributions
using Plots

# verify that A is a stochastic matrix, up to a given tolerance
function check_stochastic(A,tol=10^-1)
  n = size(A,1);
  rerr = sum(A;dims=2) .- ones(n);
  ntrue = (A .>= 0) == trues(n,n)
  return (norm(rerr,2) < tol) && ntrue, rerr, ntrue
end

# gradient flow objective function
function 𝒥(P,R,Λ)
  Γ = P * Λ * pinv(P); Δ = Γ - (R .* R);
  (1/2) * sqrt(norm(Δ,2))
end

# Lie Bracket helper
lb = (M,N) -> M*N - N*M;

# differential system describing the gradient flow on 𝒢l(n) × ℝ(n,n) -- comes from the gradient of 𝒥(P,R) with imposed constraints
# P and R are stored together ("hcat") in u
function f!(du,u,p,t)
  n = p.n; Λ = p.Λ; P = u[1:n,:]; R = u[n+1:end,:];
  Γ = P * Λ * pinv(P); Δ = Γ - (R .* R);
  du[1:n,:] = lb(transpose(Γ),Δ) * transpose(pinv(P)); du[n+1:end,:] = 2 * Δ .* R;
end

# try to find a (local) gradient-flow solution for a given set of eigenvalues λ
function Δ𝒥RStIEP(λ;tol=10^-1,maxtrials=1000,tspan=(0.0,10.0))
  n = length(λ); Λ = diagm(λ); p = (; :n => n, :Λ => Λ);

  𝒥sampG = [[] for _=1:Threads.nthreads()]; MsampG = [[] for _=1:Threads.nthreads()];

  Threads.@threads for _=1:maxtrials
    P₀ = rand(n,n); R₀ = rand(n,n); u₀ = [P₀; R₀]; # initialization
    prob = ODEProblem(f!,u₀,tspan,p); sol = solve(prob); # solve the ode
    P̂ = (sol.u[end])[1:n,:]; R̂ = (sol.u[end])[n+1:end,:]; # extract the "converged" P̂,R̂
    M̂ = P̂*Λ*pinv(P̂); 𝒥c = 𝒥(P̂,R̂,Λ); # construct the possible M̂s, compute objective function
    if check_stochastic(M̂,tol)[1]
      push!(MsampG[Threads.threadid()],M̂); push!(𝒥sampG[Threads.threadid()],𝒥c)
    end
  end

  return collect(Iterators.flatten(𝒥sampG)), collect(Iterators.flatten(MsampG))
end
##

##
λ = [0.95,0.28]; tol=(10^-1); tspan=(0.0,1000.0);
𝒥samp, Msamp = Δ𝒥RStIEP(λ;tol=tol,tspan=tspan,maxtrials=500)
##

##
λ = [0.9702,0.7066,0.5827]; tol=(10^-1); tspan=(0.0,1000.0);
𝒥samp, Msamp = Δ𝒥RStIEP(λ;tol=tol,tspan=tspan,maxtrials=250)
##

##
trial = rand(1:length(Msamp)); M = Msamp[trial]
display(eigvals(Msamp[trial]) - sort(λ));
check_stochastic(M,tol)
##