using Flux, Plots, Random, Statistics
using Flux: @epochs
accuracy(x, y) = Flux.onecold(x) .== Flux.onecold(y);

# choose a starting 1-dimensional function to model of the general form y = f(x)
# where y is the output and x is the input

function sqr(x)
    return x^2 
end

# real data samples 
function gen_real_samples(n=100)
    # Generate random inputs between -0.5 and 0.5
    X1 = rand(n) .- 0.5

    # compute their outputs
    X2 = X1.^2 
    X1 = reshape(X1, (n,1))
    X2 = reshape(X2, (n,1))
    X = [X1 X2]

    # include labels
    y = ones((n,1))

    return X, y
end

function gen_fake_samples(n=100)
    # Generate random inputs and outputs between -1 and 1
    X1 = -1 .+ rand(n) .* 2
    X2 = -1 .+ rand(n) .* 2

    X1 = reshape(X1, (n,1))
    X2 = reshape(X2, (n,1))
    X = [X1 X2]

    # include labels
    y = zeros((n,1))

    return X, y
end

trueX, truey = gen_real_samples()
fakeX, fakey = gen_fake_samples()

data = [(trueX[i,:], truey[i]) for i ∈ 1:length(truey)]
append!(data, [(fakeX[i,:], fakey[i]) for i ∈ 1:length(fakey)])

train = data[1:150]
test = data[151:end]

function define_discriminator(n_inputs=2)  
    disc = Chain(
        x-> reshape(x, n_inputs, :),
        Dense(n_inputs, 25, relu),
        Dense(25, 1, sigmoid)
    )
    return disc
end
function define_generator(latent_dim, n_outputs=2)
    gen = Chain(
        Dense(latent_dim, 15, relu),
        Dense(15, n_outputs, leakyrelu)
    )

    return gen
end

    
function bce(ŷ, y)
    neg_abs = -abs.(ŷ)
    mean(relu.(ŷ) .- ŷ .* y .+ log.(1 .+ exp.(neg_abs)))
end

disc = define_discriminator()

disc_loss(x, y) = bce(disc(x),y)

opt_disc = ADAM(1e-3)

@epochs 100 Flux.train!(disc_loss, train, opt_disc)
            # cb = [()->@show accuracy(disc, test)])

