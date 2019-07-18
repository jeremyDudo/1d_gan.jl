using Flux, Plots, Random



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
function gen_samples(n=100)
    # Generate random inputs between -0.5 and 0.5
    X1 = randn(n) .- 0.5

    # compute their outputs
    X2 = X1.^2 
    X1 = reshape(X1, (n,1))
    X2 = reshape(X2, (n,1))
    return [X1 X2]
end

data = gen_samples()

scatter(data[:,1], data[:,2])



function define_discriminator(n_inputs=2)
    model = Chain(
        Dense(25, relu),
        Dense(1, sigmoid)
    )
    loss(x,y) = binary
end