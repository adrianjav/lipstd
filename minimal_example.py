from lipstd.scaler import LipschitzScaler
from lipstd.likelihoods import *

# Declare likelihood
print("Likelihood: 'gamma', 'normal', 'categorical(3)', 'categorical(2)*', 'bernoulli*', 'poisson*'")
likelihood = get_likelihood('gamma', 'normal', 'categorical(3)', 'categorical(2)*', 'bernoulli*', 'poisson*')

# Create some synthetic data
params = torch.randn([likelihood.num_params])
for i in [0, 7, 9, 11, 13]:
    params[i] += 1  # to ensure that the Gammas have at least mode 1

likelihood.eval()
data = likelihood(*params).sample([1000])
likelihood.train()

# Apply Lipschitz standardization
scaler = LipschitzScaler(likelihood, 1 / 0.001, verbose=True)
scaler.fit(data)

# Declare the neural network with input transformation
net = nn.Sequential(
    likelihood.InputScaler,
    nn.Linear(likelihood.domain_dim, 100), nn.Tanh(),
    nn.Linear(100, likelihood.num_params)
)

params = net(data)
p_x = likelihood(*params.unbind(dim=-1))
log_evidence = p_x.log_prob(data).unbind(dim=-1)  # Data is properly transformed here in the background
print('(Transformed) Log-evidence:\t', [float(f'{l_i.mean(dim=0):.2f}') for l_i in log_evidence])

net.eval()  # To obtain the original likelihood model

params = net(data)
p_x = likelihood(*params.unbind(dim=-1))
log_evidence = p_x.log_prob(data).unbind(dim=-1)
print('(Original) Log-evidence:\t', [float(f'{l_i.mean(dim=0):.2f}') for l_i in log_evidence])
