using Plots
using SparseArrays
using Profile
using Random

using("SNN_abs.jl")

abstract type CpuIzhNetwork <: IzhNetwork end

# AboAmmar quick elementwise row multiplication
# Found here: https://stackoverflow.com/questions/48460875/vector-matrix-element-wise-multiplication-by-rows-in-julia-efficiently

function pre_trace_copier(pre_trace::Vector{Float32}, firings::Vector{Bool})
    side = length(pre_trace)
    M = zeros(side, side)

    @simd for i in 1:length(firings)
        @inbounds if firings[i]
            M[i, :] = pre_trace
        end
    end
    M
end

function post_trace_copier(post_trace::Vector{Float32}, firings::Vector{Bool})
    side = length(post_trace)
    M = zeros(side, side)

    @simd for j in 1:length(firings)
        @inbounds if firings[j]
            M[:, j] = post_trace
        end
    end
    M
end

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


function step_reward(reward::Reward, reward_injection::AbstractFloat)
    Reward(reward.reward - reward.reward/reward.decay + reward_injection, reward.decay)
end

mutable struct EligibilityTrace
    # for speed the inhibitory-ness of junctions must be stored here within the constants

    # vectors to keep track of traces, typically initialized at 0
    pre_trace::Vector{Float32}
    post_trace::Vector{Float32}
    e_trace::Matrix{Float32}

    # Parameters for pre/post incrementing and decay
    # 
    const pre_increment::AbstractFloat
    const post_increment::AbstractFloat

    # Constant to multiply junction traces by when updating the eligibility trace
    # Should typically be negative for inhibitory junctions
    const constants::Matrix{Float32}

    # Decay parameters
    const pre_decay::AbstractFloat
    const post_decay::AbstractFloat
    const e_decay::AbstractFloat
end


function step_trace!(trace::EligibilityTrace, firings::Vector{Bool})
    len_pre = length(trace.pre_trace)
    len_post = length(trace.post_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment


    # restructured logic
    len_post = length(trace.post_trace)
    len_pre = length(trace.pre_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment

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

            if firings[i]
                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.pre_trace[j]
            end

            # And if pre-synaptic neuron j spikes, we add the trace for the 
            # post-synaptic neuron to the eligibility trace

            if firings[j]
                trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.post_trace[i]
            end

            # each trace will decay according to the decay parameter
            

        end
    end

    trace.e_trace = trace.e_trace .- trace.e_trace/trace.e_decay
end

function step_trace!(trace::EligibilityTrace, firings::Vector{Bool}, mask::AbstractMatrix{Bool})

    len_post = length(trace.post_trace)
    len_pre = length(trace.pre_trace)

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment

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

                if firings[i]
                    trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.pre_trace[j]
                end

                # And if pre-synaptic neuron j spikes, we add the trace for the 
                # post-synaptic neuron to the eligibility trace

                if firings[j]
                    trace.e_trace[i, j] = trace.e_trace[i, j] + trace.constants[i,j]*trace.post_trace[i]
                end

                # each trace will decay according to the decay parameter
                
            end

        end
    end

    # each trace will decay according to the decay parameter
    trace.e_trace = trace.e_trace - trace.e_trace/trace.e_decay
end


function step_trace!(trace::EligibilityTrace, firings::Vector{Bool}, mask::SparseMatrixCSC{Bool, <:Integer})
    # Sparse masking not currently recommended, but kept because it was written

    trace.pre_trace = trace.pre_trace - trace.pre_trace/trace.pre_decay + firings * trace.pre_increment
    trace.post_trace = trace.post_trace - trace.post_trace/trace.post_decay + firings * trace.post_increment

    # restructured logic
    #
    # Pre-synaptic input (second index j, or column index j, of matrix)
    #     |
    #     v
    # . . . . .
    # . . . . . --> Post-synaptic output (first index i or row index i)
    # . . . . .

    #firings_s = firings
    addable_pre_trace = pre_trace_copier(trace.pre_trace, firings) #trace.pre_trace * firings_s'
    addable_post_trace = post_trace_copier(trace.post_trace, firings) #firings_s * trace.post_trace'
    e_trace_delta = (addable_post_trace+addable_pre_trace) .* trace.constants 
    trace.e_trace = trace.e_trace + e_trace_delta .* mask - trace.e_trace/trace.e_decay

end

function weight_update(trace::EligibilityTrace, reward::Reward)
    return reward.reward * trace.e_trace 
end

# network structures, see Izhikevich simple model and STDP papers

mutable struct CpuUnmaskedIzhNetwork <: CpuIzhNetwork
    
    # number of neurons
    const N::Integer

    # time scale recovery parameter
    const a::Vector{Float32}

    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    const b::Vector{Float32}

    # post-spike reset value of membrane potential v
    const c::Vector{Float32}

    # post-spike reset of recovery variable u
    const d::Vector{Float32}

    # membrane potential and recovery variable, used in Izhikevich system of equations
    v::Vector{Float32}
    u::Vector{Float32}

    # synaptic weights
    S::Matrix{Float32}

    # bounds used for clamping, UB should generally be 0 for inhibitory networks
    # LB should be 0 for excitatory networks
    S_ub::Matrix{Float32}
    S_lb::Matrix{Float32}


    # boolean of is-fired
    fired::Vector{Bool}

    function CpuUnmaskedIzhNetwork(N::Integer, a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32}, d::Vector{Float32}, v::Vector{Float32}, u::Vector{Float32}, S::Matrix{Float32}, S_ub::Matrix{Float32}, S_lb::Matrix{Float32}, fired::AbstractVector{Bool})
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

        return new(N, a, b, c, d, v, u, S, S_ub, S_lb, fired)
    end
end


mutable struct CpuMaskedIzhNetwork <: CpuIzhNetwork
    
    # number of neurons
    const N::Integer
    # time scale recovery parameter
    const a::Vector{Float32}
    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    const b::Vector{Float32}
    # post-spike reset value of membrane potential v
    const c::Vector{Float32}
    # post-spike reset of recovery variable u
    const d::Vector{Float32}

    # membrane poential and recovery variable, used in Izhikevich system of equations
    v::Vector{Float32}
    u::Vector{Float32}

    # synaptic weights
    S::Union{Matrix{Float32}, SparseMatrixCSC{Float32, <:Integer}}

    # bounds used for clamping, UB should generally be 0 for inhibitory networks
    # LB should be 0 for excitatory networks
    S_ub::Matrix{Float32}
    S_lb::Matrix{Float32}

    # mask
    mask::Union{AbstractMatrix{Bool}, SparseMatrixCSC{Bool, <:Integer}}

    # boolean of is-fired
    fired::Vector{Bool}

    function CpuMaskedIzhNetwork(N::Integer, a::Vector{Float32}, b::Vector{Float32}, c::Vector{Float32}, d::Vector{Float32}, v::Vector{Float32}, u::Vector{Float32}, S::Union{Matrix{Float32}, SparseMatrixCSC{Float32, <:Integer}}, S_ub::Matrix{Float32}, S_lb::Matrix{Float32}, mask::Union{AbstractMatrix{Bool}, SparseMatrixCSC{Bool, <:Integer}}, fired::AbstractVector{Bool})
        @assert length(a) == N
        @assert length(b) == N
        @assert length(c) == N
        @assert length(d) == N
        @assert length(v) == N
        @assert length(u) == N
        @assert size(S) == (N, N)
        @assert size(mask) == (N, N)
        @assert length(fired) == N

        return new(N, a, b, c, d, v, u, S, S_ub, S_lb, mask, fired)
    end


end


function step_network!(in_voltage::Vector{Float32}, network::CpuIzhNetwork)
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