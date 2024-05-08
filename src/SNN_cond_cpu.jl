const TAU_A::Float16 = 5.0
const TAU_B::Float16 = 150.0
const TAU_C::Float16 = 6.0
const TAU_D::Float16 = 150.0

const ZETA::Float16 = 0.1

abstract type CpuConductanceIzhNetwork <: IzhNetwork end

mutable struct CpuUnmaskedConductanceIzhNetwork{T<:AbstractFloat} <: CpuIzhNetwork
        
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

    # conductances for receptors
    g_a::Vector{T} # TODO: determine orders of magnitude (then swap to 64 if necessary)
    g_b::Vector{T}
    g_c::Vector{T}
    g_d::Vector{T}

    function CpuUnmaskedConductanceIzhNetwork(N::Integer, 
                                a::Vector{T}, 
                                b::Vector{T}, 
                                c::Vector{T}, 
                                d::Vector{T}, 
                                v::Vector{T}, 
                                u::Vector{T}, 
                                S::Matrix{T}, 
                                S_ub::Matrix{T}, 
                                S_lb::Matrix{T}, 
                                fired::AbstractVector{Bool},
                                g_a::Vector{T},
                                g_b::Vector{T},
                                g_c::Vector{T},
                                g_d::Vector{T}) where T <: AbstractFloat
        @assert length(a) == N
        @assert length(b) == N
        @assert length(c) == N
        @assert length(d) == N
        @assert length(v) == N
        @assert length(u) == N
        @assert length(g_a) == length(g_b) == length(g_c) == length(g_d) == N
        @assert size(S) == (N, N)
        @assert size(S_lb) == (N, N)
        @assert size(S_ub) == (N, N)
        @assert length(fired) == N

        return new{T}(N, a, b, c, d, v, u, S, S_ub, S_lb, fired, g_a, g_b, g_c, g_d) 
    end
end




function step_network!(in_voltage::Vector{<:AbstractFloat}, network::CpuConductanceIzhNetwork)
    network.fired = network.v .>= 30

    # reset voltages to c parameter values
    network.v[network.fired] .= network.c[network.fired]

    # update recovery parameter u
    network.u[network.fired] .= network.u[network.fired] + network.d[network.fired]

    # calculate new input voltages given presently firing neurons
    in_voltage = in_voltage + (network.S * network.fired)

    # Calculate total synaptic currents
    I_syn::Vector{T} = calc_synaptic_current(network)

    # update voltages (twice for stability)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage - I_syn)
    network.v = network.v + 0.5*(0.04*(network.v .^ 2) + 5*network.v .+ 140 - network.u + in_voltage - I_syn)
    network.v = min.(network.v, 30)

    # update recovery parameter u
    network.u = network.u + network.a .* (network.b .* network.v - network.u)
    
    return network
end


function calc_synaptic_current(network::CpuConductanceIzhNetwork)
    # Calculate the total synaptic current (I_syn) of the ith neuron
    I_syn::Vector{T} = network.g_a .* (network.v .- 0)
                    .+ network.g_b .* (((network.v .+ 80) / 60) .^ 2) / (1 .+ ((network.v .+ 80) / 60) .^2) .* (network.v .- 0)
                    .+ network.g_c .* (network.v .+ 70)
                    .+ network.g_d .* (network.v .+ 90)
    return I_syn
end

function update_g_a!(network::CpuIzhNetwork)
    network.g_a += -(network.g_a / TAU_A) + network.S * network.fired' * ZETA
end

function update_g_b!(network::CpuIzhNetwork)
    # Similar to update_g_a()
    network.g_b += -(network.g_b / TAU_B) + network.S * network.fired' * ZETA
end

function update_g_c!(network::CpuIzhNetwork)
    # TODO: Consider initializing "ones(N, N)" in struct -- when confirmed correct functionality 
    network.g_c += -(network.g_c / TAU_C) + ones(network.N, network.N) * network.fired' * ZETA
end

function update_g_d!(network::CpuIzhNetwork)
    # Similar to update_g_c()
    network.g_d += -(network.g_d / TAU_D) + ones(network.N, network.N) * network.fired' * ZETA
end

function step_trace!(trace::CpuEligibilityTrace, network::CpuUnmaskedConductanceIzhNetwork)

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


