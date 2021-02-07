import torch
from scipy.optimize import minimize_scalar

from .likelihoods import LikelihoodList, LikelihoodFlatten

# TODO: more general
# TODO: typing
# TODO: tests
# TODO: documentation


class LipschitzScaler(object):
    def __init__(self, likelihood, goal_smoothness, verbose=False):
        self.likelihood = likelihood
        self.goal = float(goal_smoothness)
        self.verbose = verbose

    def fit(self, data):
        def fit_single(dist, data, goal, index):
            if dist.is_discrete:
                return

            hessian = None

            old_scales = dist._scale
            dist._scale = torch.tensor([1.], device=old_scales.device)
            dist._domain_size = 1

            def step(omega):
                nonlocal hessian
                dist.scale = torch.tensor(omega).float().exp()
                lipschitz, hessian = dist.compute_lipschitz(data, hessian)
                # print(dist.scale, lipschitz, sum(lipschitz))
                return (sum(lipschitz).item() - goal) ** 2

            result = minimize_scalar(step, method='brent')
            assert result.success
            scale = torch.tensor(result.x).exp()

            dist._domain_size = old_scales.size(-1)
            dist.scale = old_scales
            dist.scale[index] = scale

            if self.verbose:
                l = dist.compute_lipschitz(data)[0]
                print(f'[{type(dist).__name__}] scale={scale:.2f} Lipschitz={sum(l).item():.2f} (goal was {goal:.2f})')

        def fit_recursive(dists, data, goal):
            if isinstance(dists, LikelihoodFlatten):
                old_value = dists.flatten
                dists.flatten = False

            pos = 0
            for d in dists:
                if isinstance(d, LikelihoodList):
                    num_dists = sum([len(x) for x in d if not x.is_discrete])
                    fit_recursive(d, data[..., pos: pos + d.domain_size], goal / num_dists)
                else:
                    for i in range(d.domain_size):
                        fit_single(d, data[..., pos + i], goal, index=i)
                pos += d.domain_size

            if isinstance(dists, LikelihoodFlatten):
                dists.flatten = old_value

        num_dists = sum([d.domain_size for d in self.likelihood if not d.is_discrete])
        fit_recursive(self.likelihood, data, self.goal / num_dists)
        return self.likelihood.scale
