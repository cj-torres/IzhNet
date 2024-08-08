module IzhNet

export CudaMaskedIzhNetwork, CudaUnmaskedIzhNetwork, CudaEligibilityTrace, CpuEligibilityTrace, CpuSimpleEligibilityTrace, CpuMaskedIzhNetwork, CpuUnmaskedIzhNetwork, Reward
export CpuConductanceIzhNetwork, CpuUnmaskedConductanceIzhNetwork, CpuMaskedConductanceIzhNetwork
export IzhParameters, CudaIzhParameters
export step_network!, step_trace!, step_reward, weight_update!, reset_network!, reset_trace!, reset_reward

include("SNN_abs.jl")
#using .IzhUtils

# include("SNN_cu.jl")
include("SNN_cpu.jl")
#include("SNN_abs.jl")

include("SNN_cond_cpu.jl")

end
