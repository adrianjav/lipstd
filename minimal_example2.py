import torch
import torch.nn as nn
from lipstd import *

# Declare likelihood
likelihood = get_likelihood('gamma', 'normal', 'categorical(3)', 'categorical(2)*', 'bernoulli*', 'poisson*')
print('Likelihoods:', likelihood, len(likelihood))
likelihood.eval()
print('Likelihoods:', likelihood, len(likelihood))
likelihood.train()

# For MNIST you could do something like this
# likelihood = get_likelihood({'name': 'bernoulli*', 'domain_size': 784})

# Create some synthetic data
params = torch.randn([likelihood.num_params])
# for i in [0, 7, 9, 11, 13]:
for i in [0, 9, 11]:
    params[i] += 1  # to ensure that the Gammas have at least mode 1

likelihood.eval()
data = likelihood(*params).sample([1000])
likelihood.train()

print('='*16)
for i in likelihood(*params):
    print(i)
print('=' * 16)

# Apply Lipschitz standardization
scaler = LipschitzScaler(likelihood, 1 / 0.001, verbose=True)
# scaler = StandardScaler(likelihood, verbose=True)
# scaler = NormalizationScaler(likelihood, verbose=True)
# scaler = InterquartileScaler(likelihood, verbose=True)
scaler.fit(data)

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

params = net(data)
p_x = likelihood(*params.unbind(dim=-1))
print('Log evidence:', p_x.log_prob(data).mean(dim=0).tolist())

net.eval()

params = net(data)
p_x = likelihood(*params.unbind(dim=-1))
print('Log evidence:', p_x.log_prob(data).mean(dim=0).tolist())
