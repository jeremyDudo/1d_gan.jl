using Flux, Plots, Random, Statistics
using Flux: @epochs

accuracy(x, y) = Flux.onecold(x) .== Flux.onecold(y);
function accuracy(x, y)
    inpt = [y[i][1] for i in 1:length(y)]
    outp = [y[i][2] for i in 1:length(y)]

    xoutp = x.(inpt)
    xoutpp = [round(xx.data[1]) for xx in xoutp]
    # print(xoutp)
    acc = mean(xoutpp .== outp)
    return acc
end
accuracy(disc, test)
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

function generate_latent_points(latent_dim, n)
    x_input = rand(latent_dim*n)
    x_input = reshape(x_input, n, latent_dim)
    return x_input
end

function gen_fake_samples(generator, latent_dim, n=100)
    x_input = generate_latent_points(latent_dim, n)

    X = generator(x_input).data
    x = X[1,:]
    y = X[2,:]
    
    return x,y
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

latent_dim = 5;
genny = define_generator(latent_dim);
fakeysX, fakeysY = gen_fake_samples(genny, latent_dim, 100)
size(fakeysX), size(fakeysY)

scatter(fakeysX,fakeysY)

trueX, truey = gen_real_samples(1000)
fakeX, fakey = gen_fake_samples(1000)

data = [(trueX[i,:], truey[i]) for i ∈ 1:length(truey)]
append!(data, [(fakeX[i,:], fakey[i]) for i ∈ 1:length(fakey)])
shuffle!(data)

train = data[1:1500]
test = data[1501:end]


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
        x-> reshape(x, latent_dim, :),
        Dense(latent_dim, 15, relu),
        Dense(15, n_outputs, leakyrelu)
    )

    return gen
end

function define_gan(generator, discriminator)
    # ideally, would make the discriminator non-trainable here, but that looks to be Keras-specific API :(
    gan = Chain(
        generator,
        discriminator
    )
    return gan
end
    
function bce(ŷ, y)
    neg_abs = -abs.(ŷ)
    mean(relu.(ŷ) .- ŷ .* y .+ log.(1 .+ exp.(neg_abs)))
end

disc = define_discriminator()

disc_loss(x, y) = bce(disc(x),y)

opt_disc = ADAM(params(disc), 0.001f0, β1 = 0.5)


@epochs 100 Flux.train!(disc_loss, train, opt_disc,
            cb = [()->@show accuracy(disc, test)])
size(train)
traininpt, trainoutp = train[:][:,1], train[:][:,2]
traininpt
trainoutp

mean([1,1,1,1,1,1,1,1,1,1,1] .== [1,1,1,1,1,0,0,0,1,1,1])

disc([0.0900755, 0.0081136])