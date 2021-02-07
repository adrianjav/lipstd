from torch import nn
from lipstd.likelihoods import *
from scipy.optimize import minimize_scalar

torch.manual_seed(2)

likelihood = get_likelihood(['gamma'], 'normal', 'one-hot-categorical(3)', ['one-hot-categorical(2)*', 'bernoulli*'], 'poisson*')
params = torch.randn([15])

for i in [0, 7, 9, 11, 13]:
    params[i] += 10

likelihood.eval()
print("['gamma'], 'normal', 'one-hot-categorical(3)', ['categorical(2)*', 'bernoulli*'], 'poisson*'")
t = likelihood(*params)
print(t)
data = t.sample([1000])
likelihood.train()
# print(d[0][0].compute_hessian(data[..., 0]))

# result, hessian = d[0][0].compute_lipschitz(data[..., 0])
# print(result[0])

from lipstd.scaler import LipschitzScaler

scaler = LipschitzScaler(likelihood, 1 / 0.001)
scaler.fit(data)

pos = 0
for i, l in enumerate(likelihood):
    if not l.is_discrete:
        print(f'{i}: {type(l).__name__} w={l.scale} L={sum(l.compute_lipschitz(data[..., pos: pos+l.domain_dim])[0]).item()}')
    pos += l.domain_dim

assert pos == likelihood.domain_dim

net = nn.Sequential(
    likelihood.InputScaler,
    nn.Linear(likelihood.domain_dim, 100), nn.Tanh(),
    nn.Linear(100, likelihood.num_params)
)

params = net(data)
list_dists = likelihood(*params.unbind(dim=-1))
for l in list_dists:
    print(str(l), l.mean.mean())

# hessian = None
#
# dist = d[1]
# data = data[..., 1]
#
# print(t[1])
#
# # dist = Normal(ensure_args=False)
# # params = torch.tensor([0, -0.5])
# # t = dist(*params)
# # data = t.sample([1000])
#
# print(d.canonical_params(*params)[1])
# print(d.scale)
# print(d.canonical_params(*params)[1])
# print(dist.canonical_params(*params[2:4]))
# print(dist.inverse_transform_params(*params[2:4]))
# print(params[2:4])
#
#
# l, h = dist.compute_lipschitz(data)
# print('lip', l, sum(l))
# print('hessian', h)
#
# print('='*10)
#
# def foo(omega):
#     global hessian
#     dist.scale = torch.tensor(omega).exp()
#     lipschitz, hessian = dist.compute_lipschitz(data, hessian)
#     print(dist.scale, lipschitz, sum(lipschitz))
#     return (sum(lipschitz).item() - 1) ** 2
#     # return (sum(norm).item() - goal * proportion[i]) ** 2
#
#
# result = minimize_scalar(foo, method='brent')
# assert result.success
# print(result)
# weight = torch.tensor(result.x).exp()
# dist.scale = weight
#
# print(weight)
# print(d.canonical_params(*params)[1])
# l = dist.compute_lipschitz(data, hessian)[0]
# print(l, sum(l))
#
#
# dist.scale = 1.
#
# print('='*10)
#
# def foo(omega):
#     dist.scale = torch.tensor(omega).exp()
#     lipschitz, _ = dist.compute_lipschitz(data)
#     print(dist.scale, lipschitz, sum(lipschitz))
#     return (sum(lipschitz).item() - 1) ** 2
#     # return (sum(norm).item() - goal * proportion[i]) ** 2
#
#
# result = minimize_scalar(foo, method='brent')
# assert result.success
# print(result)
# weight = torch.tensor(result.x).exp()
# dist.scale = weight
#
# print(d.canonical_params(*params)[1])
# l = dist.compute_lipschitz(data)[0]
# print(weight, l, sum(l))
