using Flux, Plots, Random, Statistics

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



disc = Chain(
    x-> reshape(x, 2, :),
    Dense(2, 25, relu),
    Dense(25, 1, sigmoid)
)



Flux.train!