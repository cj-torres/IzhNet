# tau: AMPA, NMDA, GABA_A, BABA_B (ms)
const TAU_A::Float16 = 5.0
const TAU_B::Float16 = 150.0
const TAU_C::Float16 = 6.0
const TAU_D::Float16 = 150.0

# constant in substitution of depression & facilitation variables
# (for conductances g)
const ZETA::Float16 = 0.1

abstract type CudaConductanceIzhNetwork <: IzhNetwork end

mutable struct CudaUnmaskedConductanceIzhNetwork{T<:AbstractFloat} <: CudaConductanceIzhNetwork
        
    # number of neurons
    const N::Integer

    # params: a, b, c, d
    params::IzhParameters{CuArray{T, 1}}

    # membrane potential and recovery variable, used in Izhikevich system of equations
    v::CuArray{T, 1}
    u::CuArray{T, 1}

    # synaptic weights
    S::CuArray{T, 2}

    # bounds used for clamping, UB should generally be 0 for inhibitory networks
    # LB should be 0 for excitatory networks
    S_ub::CuArray{T, 2}
    S_lb::CuArray{T, 2}


    # boolean of is-fired
    fired::CuArray{Bool, 1}

    # conductances for receptors
    g_a::CuArray{T, 1}
    g_b::CuArray{T, 1}
    g_c::CuArray{T, 1}
    g_d::CuArray{T, 1}

    # for g updates
    ones_matrix::CuArray{T, 2}


    function CudaUnmaskedConductanceIzhNetwork(N::Integer, 
                                params::IzhParameters{CuArray{T, 1}},
                                v::CuArray{T, 1}, 
                                u::CuArray{T, 1}, 
                                S::CuArray{T, 2}, 
                                S_ub::CuArray{T, 2}, 
                                S_lb::CuArray{T, 2}, 
                                fired::CuArray{Bool, 1},
                                g_a::CuArray{T, 1},
                                g_b::CuArray{T, 1},
                                g_c::CuArray{T, 1},
                                g_d::CuArray{T, 1}) where T <: AbstractFloat
        @assert length(params.a) == N
        @assert length(v) == length(u) == N
        @assert length(g_a) == length(g_b) == length(g_c) == length(g_d) == N
        @assert size(S) == size(S_lb) == size(S_ub) == (N, N)
        @assert length(fired) == N

        ones_matrix = ones(T, N, N)
        return new{T}(N, params, v, u, S, S_ub, S_lb, fired, g_a, g_b, g_c, g_d, ones_matrix) 
    end
end

mutable struct CudaMaskedConductanceIzhNetwork{T<:AbstractFloat} <: CudaConductanceIzhNetwork
    const N::Integer
    params::IzhParameters{CuArray{T, 1}}
    v::CuArray{T, 1}
    u::CuArray{T, 1}
    S::CuArray{T, 2}
    S_ub::CuArray{T, 2}
    S_lb::CuArray{T, 2}
    mask::CuArray{Bool, 2}
    fired::CuArray{Bool, 1}
    g_a::CuArray{T, 1}
    g_b::CuArray{T, 1}
    g_c::CuArray{T, 1}
    g_d::CuArray{T, 1}
    ones_matrix::CuArray{T, 2}

    function CudaMaskedConductanceIzhNetwork(N::Integer, 
                                params::IzhParameters{CuArray{T, 1}},
                                v::CuArray{T, 1}, 
                                u::CuArray{T, 1}, 
                                S::CuArray{T, 2}, 
                                S_ub::CuArray{T, 2}, 
                                S_lb::CuArray{T, 2}, 
                                mask::CuArray{Bool, 2},
                                fired::CuArray{Bool, 1},
                                g_a::CuArray{T, 1},
                                g_b::CuArray{T, 1},
                                g_c::CuArray{T, 1},
                                g_d::CuArray{T, 1}) where T <: AbstractFloat
        @assert length(params.a) == N
        @assert length(v) == length(u) == N
        @assert length(g_a) == length(g_b) == length(g_c) == length(g_d) == N
        @assert size(S) == size(S_lb) == size(S_ub) == size(mask) == (N, N)
        @assert length(fired) == N

        ones_matrix = ones(T, N, N)
        return new{T}(N, params, v, u, S, S_ub, S_lb, mask, fired, g_a, g_b, g_c, g_d, ones_matrix) 
    end
end

function step_network!(in_voltage::CuArray{<:AbstractFloat}, network::CudaConductanceIzhNetwork)
    network.fired = network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.params.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] + network.params.d[network.fired]
    # TODO: network.u[network.fired] .+= network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    # in_voltage = in_voltage + (network.S * network.fired)

    # Calculate total synaptic currents
    I_syn = calc_synaptic_current(network)

    # update voltages (twice for stability)
    network.v += 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage - I_syn)
    network.v += 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage - I_syn)
    network.v = min.(network.v, 30)

    # update recovery parameter u
    network.u += network.params.a .* (network.params.b .* network.v - network.u)

    update_g_a!(network)
    update_g_b!(network)
    update_g_c!(network)
    update_g_d!(network)
    
    return network
end


function calc_synaptic_current(network::CudaConductanceIzhNetwork)
    # Calculate the total synaptic current (I_syn) of the ith neuron
    I_syn = network.g_a .* (network.v .- 0)
                    .+ network.g_b .* (((network.v .+ 80) / 60) .^ 2) / (1 .+ ((network.v .+ 80) / 60) .^2) .* (network.v .- 0)
                    .+ network.g_c .* (network.v .+ 70)
                    .+ network.g_d .* (network.v .+ 90)
    return I_syn
end

function update_g_a!(network::CudaConductanceIzhNetwork)
    # print("g updated")
    network.g_a += -(network.g_a / TAU_A) + network.S * network.fired * ZETA
end

function update_g_b!(network::CudaConductanceIzhNetwork)
    # Similar to update_g_a()
    network.g_b += -(network.g_b / TAU_B) + network.S * network.fired * ZETA
end

function update_g_c!(network::CudaConductanceIzhNetwork)
    # TODO: Consider initializing "ones(N, N)" in struct -- when confirmed correct functionality 
    network.g_c += -(network.g_c / TAU_C) + network.ones_matrix * network.fired * ZETA
end

function update_g_d!(network::CudaConductanceIzhNetwork)
    # Similar to update_g_c()
    network.g_d += -(network.g_d / TAU_D) + network.ones_matrix * network.fired * ZETA
end

function step_trace!(trace::CudaEligibilityTrace, network::CudaMaskedConductanceIzhNetwork)
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


function step_trace!(trace::CudaEligibilityTrace, network::CudaUnmaskedConductanceIzhNetwork)
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

function weight_update!(network::CudaConductanceIzhNetwork, trace::CudaEligibilityTrace, reward::Reward)
    network.S = min.(max.(network.S + reward.reward * trace.e_trace, network.S_lb), network.S_ub)
    return network
end

function reset_network!(network::CudaConductanceIzhNetwork)
    network.v = network.v .* 0 .- 65.0
    network.u = network.params.b .* network.v
end

function reset_trace!(trace::CudaEligibilityTrace)
    trace.pre_trace = trace.pre_trace * 0
    trace.post_trace = trace.post_trace * 0
    trace.e_trace = trace.e_trace * 0
end