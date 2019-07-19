using Flux, Plots, Random, Statistics



# choose a starting 1-dimensional function to model of the general form y = f(x)
# where y is the output and x is the input

function sqr(x)
    return x^2 
end

# to visualize the function
inpt = -0.5:0.1:0.5
outp = sqr.(inpt)

plot(inpt, outp)


# real data samples 
function gen_real_samples(n=100)
    # Generate random inputs between -0.5 and 0.5
    X1 = randn(n) .- 0.5

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
    X1 = -1 .+ randn(n) .* 2
    X2 = -1 .+ randn(n) .* 2

    X1 = reshape(X1, (n,1))
    X2 = reshape(X2, (n,1))
    X = [X1 X2]

    # include labels
    y = zeros((n,1))

    return X, y
end

dataX, datay = gen_real_samples()
size(dataX), size(datay)
scatter(data[:,1], data[:,2])

# binary cross entropy
function bce(ŷ, y)
    neg_abs = -abs.(ŷ)
    mean(relu.(ŷ) .- ŷ .* y .+ log.(1 .+ exp.(neg_abs)))
  end

discriminator = Chain(
    x->reshape(x, 2, :),
    Dense(2, 25, relu),
    Dense(25,25,relu),
    Dense(25, 1, sigmoid)
)

disc_loss(x, y) = bce(discriminator(x),y)

opt_disc = ADAM(params(discriminator), 0.001f0, β1 = 0.5)

function train_discriminator(n_epochs=1000, n_batch=500)
    half_batch = Int(n_batch/2)

    for i ∈ 1:n_epochs

        X_real, y_real = gen_real_samples(half_batch)

        disc_real = discriminator(X_real)

        disc_real_loss = bce(disc_real, y_real)

        X_fake, y_fake = gen_fake_samples(half_batch)

        disc_fake = discriminator(X_fake)

        disc_fake_loss = bce(disc_fake, y_fake)

        disc_loss = disc_real_loss + disc_fake_loss

        Flux.back!(disc_loss)

        opt_disc()
        if i%50 == 0
            println("Epoch: $(i): | Discrimenator Loss: $(disc_loss.data)")
        end
    end
end
discriminator(dataX)
disc_loss(dataX, datay)
train_discriminator()