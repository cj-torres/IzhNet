#module IzhUtils

abstract type IzhNetwork end

struct Reward{T<:AbstractFloat}
    # i.e. "dopamine"

    # amount of "reward" present in the system
    reward::T

    # constant decay parameter
    decay::T
end

mutable struct BidirectionalConnection
    forward::Matrix{<:AbstractFloat}
    forward_mask::AbstractMatrix{Bool}

    backward::Matrix{<:AbstractFloat}
    backward_mask::AbstractMatrix{Bool}
end


mutable struct IzhSuperNetwork <: IzhNetwork
    nodes::Vector{IzhNetwork}
    connections::Dict{Tuple{Int, Int}, BidirectionalConnection}
end


struct IzhParameters{T<:AbstractArray{<:AbstractFloat}}
    # time scale recovery parameter
    a::T
    # sensitivty to sub-threshold membrane fluctuations (greater values couple v and u)
    b::T
    # post-spike reset value of membrane potential v
    c::T
    # post-spike reset of recovery variable u
    d::T

    function IzhParameters(a::T, b::T, c::T, d::T) where T <: AbstractArray{<:AbstractFloat}
        @assert length(a) == length(b) == length(c) == length(d)
        new{T}(a, b, c, d)
    end
end


function step_reward(reward::Reward, reward_injection::AbstractFloat)
    Reward(reward.reward - (reward.reward/reward.decay) + reward_injection, reward.decay)
end

function reset_reward(reward::Reward)
    Reward(reward.reward*0, reward.decay)
end


# Functions for returning parameters based on presets

function StandardSpikingParams()
    
end

function SimpleExcitatoryParams()
    
end

function SimpleInhibitoryParams()
    
end

function StandardChatteringParams()
    
end

function StandardResonantParams()
    
end

#end