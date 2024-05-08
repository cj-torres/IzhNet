using Plots
using Random

#include("SNN_abs.jl")
# using .IzhUtils

# tau: AMPA, NMDA, GABA_A, BABA_B (ms)
const TAU_A::Float16 = 5.0
const TAU_B::Float16 = 150.0
const TAU_C::Float16 = 6.0
const TAU_D::Float16 = 150.0

# constant in substitution of depression & facilitation variables
# (for conductances g)
const ZETA::Float16 = 0.1

abstract type CpuIzhNetwork <: IzhNetwork end

# Implement SSP?

# reward functionality
# Thanks be to Quintana, Perez-Pena, and Galindo (2022) for the following algorithm

#struct Reward
#    # i.e. "dopamine"
#
#    # amount of "reward" present in the system
#    reward::AbstractFloat
#
#    # constant decay parameter
#    decay::AbstractFloat
#end

mutable struct CpuEligibilityTrace{T<:AbstractFloat}
    # for speed the inhibitory-ness of junctions must be stored here within the constants

    # vectors to keep track of traces, typically initialized at 0
    pre_trace::Vector{T}
    post_trace::Vector{T}
    e_trace::Matrix{T}

    # Parameters for pre/post incrementing and decay
    # 
    const pre_increment::T
    const post_increment::T

    # Constant to multiply junction traces by when updating the eligibility trace
    # Should typically be negative for inhibitory junctions
    const constants::Matrix{T}

    # Decay parameters
    const pre_decay::T
    const post_decay::T
    const e_decay::T
end


# network structures, see Izhikevich simple model and STDP papers

mutable struct CpuUnmaskedIzhNetwork{T<:AbstractFloat} <: CpuIzhNetwork
    
    # number of neurons
    const N::Integer

    # time scale recovery parameter
    const a::Vector{T}

    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    const b::Vector{T}

    # post-spike reset value of membrane potential v
    const c::Vector{T}

    # post-spike reset of recovery variable u
    const d::Vector{T}

    # membrane potential and recovery variable, used in Izhikevich system of equations
    v::Vector{T}
    u::Vector{T}

    # synaptic weights
    S::Matrix{T}

    # bounds used for clamping, UB should generally be 0 for inhibitory networks
    # LB should be 0 for excitatory networks
    S_ub::Matrix{T}
    S_lb::Matrix{T}


    # boolean of is-fired
    fired::Vector{Bool}

    function CpuUnmaskedIzhNetwork(N::Integer, 
                                a::Vector{T}, 
                                b::Vector{T}, 
                                c::Vector{T}, 
                                d::Vector{T}, 
                                v::Vector{T}, 
                                u::Vector{T}, 
                                S::Matrix{T}, 
                                S_ub::Matrix{T}, 
                                S_lb::Matrix{T}, 
                                fired::AbstractVector{Bool}) where T <: AbstractFloat
        @assert length(a) == N
        @assert length(b) == N
        @assert length(c) == N
        @assert length(d) == N
        @assert length(v) == N
        @assert length(u) == N
        @assert size(S) == (N, N)
        @assert size(S_lb) == (N, N)
        @assert size(S_ub) == (N, N)
        @assert length(fired) == N

        return new{T}(N, a, b, c, d, v, u, S, S_ub, S_lb, fired, g_a, g_b, g_c, g_d) 
    end
end


mutable struct CpuMaskedIzhNetwork{T<:AbstractFloat} <: CpuIzhNetwork
    
    # number of neurons
    const N::Integer
    # time scale recovery parameter
    const a::Vector{T}
    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    const b::Vector{T}
    # post-spike reset value of membrane potential v
    const c::Vector{T}
    # post-spike reset of recovery variable u
    const d::Vector{T}

    # membrane poential and recovery variable, used in Izhikevich system of equations
    v::Vector{T}
    u::Vector{T}

    # synaptic weights
    S::Matrix{T}

    # bounds used for clamping, UB should generally be 0 for inhibitory networks
    # LB should be 0 for excitatory networks
    S_ub::Matrix{T}
    S_lb::Matrix{T}

    # mask
    mask::Matrix{Bool}

    # boolean of is-fired
    fired::Vector{Bool}

    function CpuMaskedIzhNetwork(N::Integer, 
                                a::Vector{T}, 
                                b::Vector{T}, 
                                c::Vector{T}, 
                                d::Vector{T}, 
                                v::Vector{T}, 
                                u::Vector{T}, 
                                S::Matrix{T}, 
                                S_ub::Matrix{T},
                                S_lb::Matrix{T}, 
                                mask::Matrix{Bool}, 
                                fired::Vector{Bool}) where T <: AbstractFloat
        @assert length(a) == N
        @assert length(b) == N
        @assert length(c) == N
        @assert length(d) == N
        @assert length(v) == N
        @assert length(u) == N
        @assert size(S) == (N, N)
        @assert size(mask) == (N, N)
        @assert length(fired) == N

        return new{T}(N, a, b, c, d, v, u, S, S_ub, S_lb, mask, fired)
    end


end


function step_network!(in_voltage::Vector{<:AbstractFloat}, network::CpuIzhNetwork)
    network.fired = network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] + network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage = in_voltage + (network.S * network.fired)

    # update voltages (twice for stability)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)
    network.v = min.(network.v, 30)

    # update recovery parameter u
    network.u = network.u + network.a .* (network.b .* network.v - network.u)
    
    return network
end

function step_trace!(trace::CpuEligibilityTrace, network::CpuUnmaskedIzhNetwork)

    # restructured logic
    len_post = length(trace.post_trace)
    len_pre = length(trace.pre_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + network.fired * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + network.fired * trace.post_increment

    @inbounds for i in 1:len_post
        @inbounds @simd for j in 1:len_pre


            # i is row index, j is column index, so...
            #
            #
            # Pre-synaptic input (indexed with j)
            #     |
            #     v
            # . . . . .
            # . . . . . --> Post-synaptic output (indexed with i)
            # . . . . .
            
            # Check if presynaptic neuron is inhibitory

            # We add the *opposite* trace given a neural spike
            # So if post-synaptic neuron i spikes, we add the trace for the 
            # pre-synaptic neuron to the eligibility trace

            if network.fired[i]
                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.pre_trace[j]
            end

            # And if pre-synaptic neuron j spikes, we add the trace for the 
            # post-synaptic neuron to the eligibility trace

            if network.fired[j]
                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.post_trace[i]
            end

            # each trace will decay according to the decay parameter
            

        end
    end

    trace.e_trace = trace.e_trace .- trace.e_trace/trace.e_decay
end

function step_trace!(trace::CpuEligibilityTrace, network::CpuMaskedIzhNetwork)

    len_post = length(trace.post_trace)
    len_pre = length(trace.pre_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + network.fired * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + network.fired * trace.post_increment

    @inbounds for i in 1:len_post
        @inbounds @simd for j in 1:len_pre


            # i is row index, j is column index, so...
            #
            #
            # Pre-synaptic input (indexed with j)
            #     |
            #     v
            # . . . . .
            # . . . . . --> Post-synaptic output (indexed with i)
            # . . . . .
            
            # Check if presynaptic neuron is inhibitory

            # Check if the neurons have a synpatic connection j -> i
            # remember wonky row/column indexing makes things "backwards"
            if mask[i,j]

                # We add the *opposite* trace given a neural spike
                # So if post-synaptic neuron i spikes, we add the trace for the 
                # pre-synaptic neuron to the eligibility trace

                if network.fired[i]
                    trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.pre_trace[j]
                end

                # And if pre-synaptic neuron j spikes, we add the trace for the 
                # post-synaptic neuron to the eligibility trace

                if network.fired[j]
                    trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.post_trace[i]
                end

                # each trace will decay according to the decay parameter
                
            end

        end
    end

    # each trace will decay according to the decay parameter
    trace.e_trace = trace.e_trace - trace.e_trace/trace.e_decay
end

function weight_update!(network::CpuIzhNetwork, trace::CpuEligibilityTrace, reward::Reward)
    network.S = min.(max.(network.S + reward.reward * trace.e_trace, network.S_lb), network.S_ub)
    return network
end

function reset_network!(network::CpuIzhNetwork)
    network.v = network.v .* 0 .- 65.0
    network.u = network.b .* network.v
end

function reset_trace!(trace::CpuEligibilityTrace)
    trace.pre_trace = trace.pre_trace * 0
    trace.post_trace = trace.post_trace * 0
    trace.e_trace = trace.e_trace * 0
end

#function step_network!(in_voltage::Vector{Float32}, network::CpuMaskedIzhNetwork)
#    network.fired = network.v .>= 30
#
#    # reset voltages to c parameter values
#    network.v[network.fired] .= network.c[network.fired]
#
#    # update recovery parameter u
#    network.u[network.fired] .= network.u[network.fired] + network.d[network.fired]
#
#    # calculate new input voltages given presently firing neurons
#    in_voltage = in_voltage + ((network.S .* network.mask) * network.fired)
#
#    # update voltages (twice for stability)
#    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)
#    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage)
#    network.v = min.(network.v, 30)
#
#    # update recovery parameter u
#    network.u = network.u + network.a .* (network.b .* network.v - network.u)
#
#end