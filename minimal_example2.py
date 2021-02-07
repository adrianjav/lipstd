from lipstd.scaler import LipschitzScaler
from lipstd.likelihoods import *

# Declare likelihood
likelihood = get_likelihood('gamma', 'normal', 'categorical(3)', 'categorical(2)*', 'bernoulli*', 'poisson*')

# Create some synthetic data
params = torch.randn([likelihood.num_params])
for i in [0, 7, 9, 11, 13]:
    params[i] += 1  # to ensure that the Gammas have at least mode 1

likelihood.eval()
data = likelihood(*params).sample([1000])
likelihood.train()

# Apply Lipschitz standardization
scaler = LipschitzScaler(likelihood, 1 / 0.001)
scaler.fit(data)

# For MNIST you could do something like this
# likelihood = LikelihoodFlatten(GammaTrick(Bernoulli(domain_size=784)))
# scaler = LipschitzScaler(likelihood, 1. / args.learning_rate)
# scaler.fit(data_loader.dataset.data.flatten(start_dim=1) / 255.)

print("Likelihood: 'gamma', 'normal', 'categorical(3)', 'categorical(2)*', 'bernoulli*', 'poisson*'")
pos = 0
for i, l in enumerate(likelihood):
    if not l.is_discrete:
        Ls = l.compute_lipschitz(data[..., pos: pos+l.domain_size])[0]
        print(f'{i}: {type(l).__name__} w={l.scale.item():.2f} L={sum(Ls).item():.2f}')
    pos += l.domain_size

net = nn.Sequential(
    likelihood.InputScaler,
    nn.Linear(likelihood.domain_size, 100), nn.Tanh(),
    nn.Linear(100, likelihood.num_params)
)

net.eval()

params = net(data)
p_x = likelihood(*params.unbind(dim=-1))
print('Log evidence:', p_x.log_prob(data).mean(dim=0).tolist())
