using CUDA
using Plots
using Random

using("SNN_abs.jl")

abstract type CudaIzhNetwork <: IzhNetwork end

struct Reward
    # i.e. "dopamine"

    # amount of "reward" present in the system
    reward::AbstractFloat

    # constant decay parameter
    decay::AbstractFloat
end

mutable struct CudaEligibilityTrace
    pre_trace::CuArray{<:AbstractFloat, 1}
    post_trace::CuArray{<:AbstractFloat, 1}
    e_trace::CuArray{<:AbstractFloat, 2}
    const pre_increment::AbstractFloat
    const post_increment::AbstractFloat
    const constants::CuArray{<:AbstractFloat, 2}
    const pre_decay::AbstractFloat
    const post_decay::AbstractFloat
    const e_decay::AbstractFloat
end


function step_reward(reward::Reward, reward_injection::AbstractFloat)
    Reward(reward.reward .- reward.reward ./ reward.decay .+ reward_injection, reward.decay)
end


function weight_update(trace::CudaEligibilityTrace, reward::Reward)
    return reward.reward * trace.e_trace 
end

function step_trace!(trace::CudaEligibilityTrace, firings::CuArray{Bool, 1}, mask::CuArray{Bool, 2})
    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ firings .* trace.pre_increment
    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ firings .* trace.post_increment

    # Vectorized update for e_trace
    # Broadcasting the firings array to match the dimensions of e_trace
    firings_row = reshape(firings, 1, :)
    firings_col = reshape(firings, :, 1)

    trace.e_trace .= trace.e_trace .+ ((trace.constants .* trace.pre_trace') .* mask) .* firings_col
    trace.e_trace .= trace.e_trace .+ ((trace.constants .* trace.post_trace) .* mask) .* firings_row

    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
end


function step_trace!(trace::CudaEligibilityTrace, firings::CuArray{Bool, 1})
    trace.pre_trace .= trace.pre_trace .- trace.pre_trace ./ trace.pre_decay .+ firings .* trace.pre_increment
    trace.post_trace .= trace.post_trace .- trace.post_trace ./ trace.post_decay .+ firings .* trace.post_increment

    # GPU-compatible way to update e_trace
    # You may need to adjust this logic based on the specifics of your model
    for i in 1:length(firings)
        if firings[i]
            trace.e_trace[i, :] .= trace.e_trace[i, :] .+ trace.constants[i, :] .* trace.pre_trace
            trace.e_trace[:, i] .= trace.e_trace[:, i] .+ trace.constants[:, i] .* trace.post_trace
        end
    end

    trace.e_trace .= trace.e_trace .- trace.e_trace ./ trace.e_decay
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

mutable struct CudaUnmaskedIzhNetwork <: CudaIzhNetwork
    const N::Integer
    const a::CuArray{<:AbstractFloat, 1}
    const b::CuArray{<:AbstractFloat, 1}
    const c::CuArray{<:AbstractFloat, 1}
    const d::CuArray{<:AbstractFloat, 1}
    v::CuArray{<:AbstractFloat, 1}
    u::CuArray{<:AbstractFloat, 1}
    S::CuArray{<:AbstractFloat, 2}
    S_ub::CuArray{<:AbstractFloat, 2}
    S_lb::CuArray{<:AbstractFloat, 2}
    fired::CuArray{Bool, 1}

    function CudaUnmaskedIzhNetwork(N::Integer, 
                                a::CuArray{<:AbstractFloat, 1}, 
                                b::CuArray{<:AbstractFloat, 1}, 
                                c::CuArray{<:AbstractFloat, 1}, 
                                d::CuArray{<:AbstractFloat, 1}, 
                                v::CuArray{<:AbstractFloat, 1}, 
                                u::CuArray{<:AbstractFloat, 1}, 
                                S::CuArray{<:AbstractFloat, 2}, 
                                S_ub::CuArray{<:AbstractFloat, 2}, 
                                S_lb::CuArray{<:AbstractFloat, 2}, 
                                fired::CuArray{Bool, 1})
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


mutable struct CudaMaskedIzhNetwork <: CudaIzhNetwork
    const N::Integer
    const a::CuArray{<:AbstractFloat, 1}
    const b::CuArray{<:AbstractFloat, 1}
    const c::CuArray{<:AbstractFloat, 1}
    const d::CuArray{<:AbstractFloat, 1}
    v::CuArray{<:AbstractFloat, 1}
    u::CuArray{<:AbstractFloat, 1}
    S::CuArray{<:AbstractFloat, 2}
    S_ub::CuArray{<:AbstractFloat, 2}
    S_lb::CuArray{<:AbstractFloat, 2}
    mask::CuArray{Bool, 2}
    fired::CuArray{Bool, 1}

    function CudaMaskedIzhNetwork(N::Integer, 
                                a::CuArray{<:AbstractFloat, 1}, 
                                b::CuArray{<:AbstractFloat, 1}, 
                                c::CuArray{<:AbstractFloat, 1}, 
                                d::CuArray{<:AbstractFloat, 1}, 
                                v::CuArray{<:AbstractFloat, 1}, 
                                u::CuArray{<:AbstractFloat, 1}, 
                                S::CuArray{<:AbstractFloat, 2}, 
                                S_ub::CuArray{<:AbstractFloat, 2}, 
                                S_lb::CuArray{<:AbstractFloat, 2}, 
                                mask::CuArray{Bool, 2}, 
                                fired::CuArray{Bool, 1})
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


function step_network!(in_voltage::CuArray{<:AbstractFloat, 1}, network::CudaUnmaskedIzhNetwork)
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
end