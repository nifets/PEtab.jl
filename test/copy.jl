using PEtab, Test

path_yaml = joinpath(
    @__DIR__, "published_models", "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"
)
prob = PEtabODEProblem(PEtabModel(path_yaml); hessian_method = :GaussNewton)
x = get_x(prob)

@testset "copy is independent of the original" begin
    prob_copy = copy(prob)
    @test prob_copy.probinfo.cache !== prob.probinfo.cache
    @test prob_copy.model_info.simulation_info.odesols !== prob.model_info.simulation_info.odesols
    @test prob_copy.nllh(x) ≈ prob.nllh(x)
    @test prob_copy.grad(x) ≈ prob.grad(x)
    @test prob_copy.hess(x) ≈ prob.hess(x)
end
