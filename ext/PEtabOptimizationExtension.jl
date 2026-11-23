module PEtabOptimizationExtension

import Catalyst: @unpack
import ComponentArrays: ComponentVector
import Optimization: OptimizationFunction, OptimizationProblem
import PEtab
import QuasiMonteCarlo: LatinHypercubeSample
import Random
import SciMLBase

function SciMLBase.OptimizationProblem(
        prob::PEtab.PEtabODEProblem; box_constraints::Bool = true
    )::OptimizationProblem
    # OptimizationFunction with PEtab.jl objective, gradient and Hessian
    _f = (u, p) -> prob.nllh(u)
    _grad! = (G, u, p) -> prob.grad!(G, u)
    _hess! = (H, u, p) -> prob.hess!(H, u)
    optf = OptimizationFunction(_f; grad = _grad!, hess = _hess!)

    # OptimizationProblem
    @unpack lower_bounds, upper_bounds = prob
    x0 = deepcopy(prob.xnominal_transformed)
    if box_constraints == false
        lower_bounds, upper_bounds = nothing, nothing
    end
    return OptimizationProblem(
        optf, x0, SciMLBase.NullParameters(); lb = lower_bounds, ub = upper_bounds
    )
end

function PEtab.calibrate_multistart(
        rng::Random.AbstractRNG, prob::PEtab.PEtabODEProblem, alg, nmultistarts;
        nprocs = 1, save_trace = false, dirsave = nothing, sample_prior = true,
        sampling_method = LatinHypercubeSample(), show_progress = false,
        init_weight = nothing, init_bias = nothing, options = NamedTuple()
    )::PEtab.PEtabMultistartResult
    return PEtab._calibrate_multistart(
        rng, prob, alg, nmultistarts, dirsave, sampling_method, options, sample_prior,
        save_trace, nprocs, show_progress, init_weight, init_bias
    )
end

function PEtab.calibrate(
        prob::PEtab.PEtabODEProblem, x::AbstractVector{<:Real},
        alg; save_trace::Bool = false, options = NamedTuple()
    )::PEtab.PEtabOptimisationResult
    x0 = PEtab._get_x_out(x, prob)
    optprob = SciMLBase.remake(OptimizationProblem(prob); u0 = deepcopy(x0))

    ftrace = Vector{Float64}(undef, 0)
    xtrace = Vector{Vector{Float64}}(undef, 0)
    callback = function (state, loss)
        if save_trace
            push!(ftrace, loss)
            push!(xtrace, collect(state.u))
        end
        return false
    end

    runtime = @elapsed sol = SciMLBase.solve(optprob, alg; callback, options...)

    xmin = PEtab._get_x_out(sol.u, prob)
    fmin = isfinite(sol.objective) ? sol.objective : NaN
    return PEtab.PEtabOptimisationResult(
        xmin, fmin, x0, nameof(typeof(alg)), sol.stats.iterations, runtime, xtrace,
        ftrace, sol.retcode, sol
    )
end

end
