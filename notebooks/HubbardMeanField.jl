### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 2264b716-a37a-11eb-1393-7f19142f6d9c
begin
	using LinearAlgebra
	using BandedMatrices
	
	using Plots; Plots.gr()
	using LaTeXStrings
end

# ╔═╡ 44c0e18e-a42b-11eb-1810-3fe2e9e81d53
begin
	# Taken from https://github.com/JuliaMatrices/BandedMatrices.jl/issues/180
	function kron(a::BandedMatrix{T}, b::BandedMatrix{V}) where {T,V}
		R = BandedMatrix{Base.promote_op(*,T,V)}(undef, (size(a,1) * size(b,1), size(a,2) * size(b,2)), (bandwidth(a,1) * size(b,1) + bandwidth(b,1), bandwidth(a,2) * size(b,2) + bandwidth(b,2)))
		m = 0
		@inbounds for j = 1:size(a,2), l = 1:size(b,2), i = 1:size(a,1)
			aij = a[i,j]
			for k = 1:size(b,1)
				R[m += 1] = aij * b[k,l]
			end
		end
		R
	end
	
	kron(A::BandedMatrix, B::BandedMatrix, C::AbstractMatrix...) = kron_(kron_(A,B), C...)
	kron(A) = A
end

# ╔═╡ 3cec498e-a37b-11eb-0b97-e33d6bec2dea
begin
	# Simulation parameters
	const N_max = 5 	# Fock space truncation
	
	# Single site operators
	const bₙ = BandedMatrix(1 => (1:N_max) .^ (1/2))
	const Iₙ = BandedMatrix(0 => ones(Float64, N_max+1))
	
	@assert bₙ' * bₙ ≈ Iₙ .* (0:N_max)
	
	"""
		H(t, μ, b̄, L)
	
	Bose-Hubbard `L`-chain mean-field Hamiltonian
	"""
	function H(t, μ, b̄, L)
		b = [kron((Iₙ for _ in 1:(i-1))..., bₙ, (Iₙ for _ in (i+1):L)...) for i in 1:L]
		I = BandedMatrix(0 => ones(Float64, (N_max+1)^L))
		
		# on-site
		h = sum(-μ * b[i]' * b[i] + 0.5 * b[i]' * b[i] .* (b[i]' * b[i] .- I) for i in 1:L)

		# mean-field hopping
		h += -t * (b[1]' * b̄ + b̄' * b[1])

		if L > 1
			# mean-field hopping
			h += -t * (b[L]' * b̄ + b̄' * b[L])

			# chain hopping
			h += -t * sum(b[i+1]' * b[i] + b[i]' * b[i+1] for i in 1:(L-1))
		end
		
		return h
	end
	
	"""
		mean_field(t, μ, b̄, L)
	
	Bose-Hubbard mean-field at T=0 (ϕ being the ground-state)
		<ϕ|b|ϕ>
	"""
	function mean_field(t, μ, b̄, L)
		ϕ = begin
			_, vecs = eigen(Matrix(H(t, μ, b̄, L)))
			vecs[:,1] # eigen-vector of the lowest eigen-value
		end
		
		# If H is huge, prefer `using KrylovKit`
		# ϕ = eigsolve(H(t, μ, b̄, L), 1, :SR)[2][1]
		
		return ϕ' * kron(bₙ, (Iₙ for _ in 2:L)...) * ϕ
	end
	
	"""
		fixedpoint(f, x0; xtol)
	
	Finds the fixed point
		x* = f(x*)
	"""
	function fixedpoint(f, x; xtol=1e-4, β=0.1)
		x0, x = Inf, x
		while norm(x - x0, Inf) > xtol
			x0, x = x, β * x + (1 - β) * f(x)
		end
		return x
	end
end;

# ╔═╡ 80506ae8-a40e-11eb-117c-6b22718cc945
begin
	# Physical parameters (units of U)
	μs = 0:0.03:4
	ts = 0:0.006:0.2
	
	L = 1 	# chain size
	
	# Parameter scan
	data = [fixedpoint(b̄ -> mean_field(t, μ, b̄, L), 1.0) for μ in μs, t in ts]
end;

# ╔═╡ 3958ae48-a37d-11eb-26f7-b7511953016e
heatmap(ts, μs, data;
	xlabel=L"t/U",
	ylabel=L"\mu/U",
	title=L"\textrm{mean-field} <b>",
	xlims=extrema(ts))

# ╔═╡ Cell order:
# ╠═2264b716-a37a-11eb-1393-7f19142f6d9c
# ╟─44c0e18e-a42b-11eb-1810-3fe2e9e81d53
# ╠═3cec498e-a37b-11eb-0b97-e33d6bec2dea
# ╠═80506ae8-a40e-11eb-117c-6b22718cc945
# ╠═3958ae48-a37d-11eb-26f7-b7511953016e
