abstract type IzhNetwork end

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