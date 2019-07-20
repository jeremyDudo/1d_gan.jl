using Flux, Plots, Random, Statistics


###############################################################################################
###################################### Sample Preparation #####################################

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
    
    y = zeros((n,1))
    return X,y
end

###################################### Example Samples #########################################
trueX, truey = gen_real_samples(1000)
fakeX, fakey = gen_fake_samples(1000)

data = [(trueX[i,:], truey[i]) for i ∈ 1:length(truey)]
append!(data, [(fakeX[i,:], fakey[i]) for i ∈ 1:length(fakey)])
shuffle!(data)

train = data[1:1500]
test = data[1501:end]

################################################################################################
######################################### Architecture #########################################

n_inputs = 2;
n_outputs = 2;
latent_dim = 20;

########################################## Generator ###########################################
gen = Chain(
    x-> reshape(x, latent_dim, :),
    Dense(latent_dim, 15, relu),
    Dense(15, n_outputs, leakyrelu)
)
######################################## Discriminator #########################################
disc = Chain(
    x-> reshape(x, n_inputs, :),
    Dense(n_inputs, 25, relu),
    Dense(25, 1, sigmoid)
)
################################################################################################

opt_gen  = ADAM(params(gen), 0.001f0, β1 = 0.5)
opt_disc = ADAM(params(disc), 0.001f0, β1 = 0.5)

############################### Helper Functions ###############################

function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

################################ Loss and Training ##############################
# binary cross entropy
function bce(ŷ, y)
    neg_abs = -abs.(ŷ)
    mean(relu.(ŷ) .- ŷ .* y .+ log.(1 .+ exp.(neg_abs)))
end

training_steps = 0
verbose_freq = 100

function train(x)
    global training_steps
  
    # z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
    realX, realy = gen_real_samples(x)
   
    zero_grad!(disc)
   
    D_real = disc(realX)
    D_real_loss = bce(D_real, realy)
  
    fakeX, fakey = gen_fake_samples(gen, latent_dim, x)

    D_fake = disc(fakeX)
    D_fake_loss = bce(D_fake, fakey)
  
    D_loss = D_real_loss + D_fake_loss
  
    Flux.back!(D_loss)
    opt_disc()
  
    zero_grad!(gen)
    
    # z = rand(dist, noise_dim, BATCH_SIZE) |> gpu
    fakeX, fakey = gen_fake_samples(gen, latent_dim, x)
    D_fake = disc(fakeX)
    G_loss = bce(D_fake, ones(x))
  
    Flux.back!(G_loss)
    opt_gen()
  
    if training_steps % verbose_freq == 0
        println("D Loss: $(D_loss.data) | G loss: $(G_loss.data)")
    end
    
    training_steps += 1
    param(0.0f0)
end

NUM_EPOCHS = 10000;

for e = 1:NUM_EPOCHS
    println("Epoch $e: ")
    train(500)
end

disc([0.002, 0.002^2])

x_fake, y_fake = gen_fake_samples(gen, latent_dim)
x_fake[:, 1]
scatter(x_fake[2, :], x_fake[1, :])
