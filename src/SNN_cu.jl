using CUDA
using Plots
using Random

#include("SNN_abs.jl")
#using .IzhUtils

abstract type CudaIzhNetwork <: IzhNetwork end

mutable struct CudaEligibilityTrace{T<:AbstractFloat}
    pre_trace::CuArray{T, 1}
    post_trace::CuArray{T, 1}
    e_trace::CuArray{T, 2}
    const pre_increment::T
    const post_increment::T
    const constants::CuArray{T, 2}
    const pre_decay::T
    const post_decay::T
    const e_decay::T
end


#function step_trace!(trace::EligibilityTrace, firings::CuArray{Bool, 1}, mask::CuArray{Bool, 2})
#    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ firings .* trace.pre_increment
#    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ firings .* trace.post_increment#
#
#    # GPU-compatible way to update e_trace with mask
#    for i in 1:length(firings)
#        if firings[i]
#            trace.e_trace[i, :] .= trace.e_trace[i, :] .+ (trace.constants[i, :] .* trace.pre_trace) .* mask[i, :]
#            trace.e_trace[:, i] .= trace.e_trace[:, i] .+ (trace.constants[:, i] .* trace.post_trace) .* mask[:, i]
#        end
#    end
#
#    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
#end

mutable struct CudaUnmaskedIzhNetwork{T<:AbstractFloat} <: CudaIzhNetwork
    const N::Integer
    const a::CuArray{T, 1}
    const b::CuArray{T, 1}
    const c::CuArray{T, 1}
    const d::CuArray{T, 1}
    v::CuArray{T, 1}
    u::CuArray{T, 1}
    S::CuArray{T, 2}
    S_ub::CuArray{T, 2}
    S_lb::CuArray{T, 2}
    fired::CuArray{Bool, 1}

    function CudaUnmaskedIzhNetwork(N::Integer, 
                                a::CuArray{T, 1}, 
                                b::CuArray{T, 1}, 
                                c::CuArray{T, 1}, 
                                d::CuArray{T, 1}, 
                                v::CuArray{T, 1}, 
                                u::CuArray{T, 1}, 
                                S::CuArray{T, 2}, 
                                S_ub::CuArray{T, 2}, 
                                S_lb::CuArray{T, 2}, 
                                fired::CuArray{Bool, 1}) where T <: AbstractFloat
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

        return new{T}(N, a, b, c, d, v, u, S, S_ub, S_lb, fired)
    end
end


mutable struct CudaMaskedIzhNetwork{T<:AbstractFloat} <: CudaIzhNetwork
    const N::Integer
    const a::CuArray{T, 1}
    const b::CuArray{T, 1}
    const c::CuArray{T, 1}
    const d::CuArray{T, 1}
    v::CuArray{T, 1}
    u::CuArray{T, 1}
    S::CuArray{T, 2}
    S_ub::CuArray{T, 2}
    S_lb::CuArray{T, 2}
    mask::CuArray{Bool, 2}
    fired::CuArray{Bool, 1}

    function CudaMaskedIzhNetwork(N::Integer, 
                                a::CuArray{T, 1}, 
                                b::CuArray{T, 1}, 
                                c::CuArray{T, 1}, 
                                d::CuArray{T, 1}, 
                                v::CuArray{T, 1}, 
                                u::CuArray{T, 1}, 
                                S::CuArray{T, 2}, 
                                S_ub::CuArray{T, 2}, 
                                S_lb::CuArray{T, 2}, 
                                mask::CuArray{Bool, 2}, 
                                fired::CuArray{Bool, 1}) where T <: AbstractFloat
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


function step_network!(in_voltage::CuArray{<:AbstractFloat, 1}, network::CudaUnmaskedIzhNetwork)
    network.fired .= network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] .+ network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage .= in_voltage .+ (network.S * network.fired)

    # Calculate total synaptic currents
    # TODO: rewrite I_syn as CuArray

    # TODO: subtract I_syn from voltages
    # update voltages (twice for stability)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)
    network.v .= min.(network.v, 30)

    # update recovery parameter u
    network.u .= network.u .+ network.a .* (network.b .* network.v .- network.u)

    return network
end


function step_network!(in_voltage::CuArray{<:AbstractFloat, 1}, network::CudaMaskedIzhNetwork)
    network.fired .= network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] .+ network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage .= in_voltage .+ (network.S * network.fired)

    # update voltages (twice for stability)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)
    network.v .= network.v .+ 0.5 * (0.04 * (network.v .^ 2) .+ 5 .* network.v .+ 140 .- network.u .+ in_voltage)
    network.v .= min.(network.v, 30)

    # update recovery parameter u
    network.u .= network.u .+ network.a .* (network.b .* network.v .- network.u)

    return network
end

function calc_synaptic_current(network::CudaIzhNetwork)
    # Calculate the total synaptic current (I_syn) of the ith neuron
    I_syn::Vector{T} = network.g_a * (network.v .- 0)
                    .+ network.g_b * (((network.v .+ 80) / .60) .^ 2) / (1 .+ ((network.v .+ 80) / 60) .^2) * (network.v .- 0)
                    .+ network.g_c * (network.v .+ 70)
                    .+ network.g_d * (network.v .+ 90)
    return I_syn
end

function update_g_a!(network::CudaIzhNetwork)
    network.g_a += -(network.g_a / TAU_A) + network.S * network.fired' * ZETA
end

function update_g_b!(network::CudaIzhNetwork)
    # Similar to update_g_a()
    network.g_b += -(network.g_b / TAU_B) + network.S * network.fired' * ZETA
end

function update_g_c!(network::CudaIzhNetwork)
    # TODO: Consider initializing "ones(N, N)" in struct -- when confirmed correct functionality 
    network.g_c += -(network.g_c / TAU_C) + ones(network.N, network.N) * network.fired' * ZETA
end

function update_g_d!(network::CudaIzhNetwork)
    # Similar to update_g_c()
    network.g_d += -(network.g_d / TAU_D) + ones(network.N, network.N) * network.fired' * ZETA
end

function step_trace!(trace::CudaEligibilityTrace, network::CudaMaskedIzhNetwork)
    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ network.fired .* trace.pre_increment
    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ network.fired .* trace.post_increment

    # Vectorized update for e_trace
    # Broadcasting the firings array to match the dimensions of e_trace
    firings_row = reshape(network.fired, 1, :)
    firings_col = reshape(network.fired, :, 1)

    trace.e_trace .= trace.e_trace .+ ((trace.constants .* trace.pre_trace') .* network.mask) .* firings_col
    trace.e_trace .= trace.e_trace .+ ((trace.constants .* trace.post_trace) .* network.mask) .* firings_row

    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
end


function step_trace!(trace::CudaEligibilityTrace, network::CudaUnmaskedIzhNetwork)
    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ network.fired .* trace.pre_increment
    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ network.fired .* trace.post_increment

    # GPU-compatible way to update e_trace
    # You may need to adjust this logic based on the specifics of your model
    for i in eachindex(network.fired)
        if network.fired[i]
            trace.e_trace[i, :] .= trace.e_trace[i, :] .+ trace.constants[i, :] .* trace.pre_trace
            trace.e_trace[:, i] .= trace.e_trace[:, i] .+ trace.constants[:, i] .* trace.post_trace
        end
    end

    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay

    return trace
end

function weight_update!(network::CudaIzhNetwork, trace::CudaEligibilityTrace, reward::Reward)
    network.S = min.(max.(network.S + reward.reward * trace.e_trace, network.S_lb), network.S_ub)
    return network
end

function reset_network!(network::CudaIzhNetwork)
    network.v = network.v .* 0 .- 65.0
    network.u = network.b .* network.v
end

function reset_trace!(trace::CudaEligibilityTrace)
    trace.pre_trace = trace.pre_trace * 0
    trace.post_trace = trace.post_trace * 0
    trace.e_trace = trace.e_trace * 0
end