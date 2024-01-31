module IzhNet

export CudaMaskedIzhNetwork, CudaUnmaskedIzhNetwork, CudaEligibilityTrace, CpuEligibilityTrace, CpuMaskedIzhNetwork, CpuUnmaskedIzhNetwork, Reward
export step_network!, step_trace!, step_reward, weight_update!

include("SNN_abs.jl")
#using .IzhUtils

include("SNN_cu.jl")
include("SNN_cpu.jl")
#include("SNN_abs.jl")


end
