import torch
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import RobustScaler

from .likelihoods import LikelihoodList, LikelihoodFlatten

# TODO: more general (e.g. continuous multivariate distributions)
# TODO: typing
# TODO: tests
# TODO: documentation


class BaseScaler(object):
    def __init__(self, likelihood, verbose=False):
        self.likelihood = likelihood  # TODO get with name list
        self.verbose = verbose

    def fit_single(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, data):
        pos = 0
        for d in self.likelihood:
            if not d.is_discrete:
                # Reset the scale and preprocess (to account for things like dequantization)
                d.scale = torch.ones_like(d.scale)
                data_d = d >> data[..., pos: pos + d.domain_size]

                scale_d = torch.empty_like(d.scale)
                for i in range(d.domain_size):
                    scale_d[i] = self.fit_single(data_d[..., i])
                    if self.verbose:
                        print(f'[x_{pos+i}] scale={scale_d[i]:.2f}')
                d.scale = scale_d
            pos += d.domain_size

        return self.likelihood


class StandardScaler(BaseScaler):
    def fit_single(self, data):
        return 1./data.std().item()


class NormalizationScaler(BaseScaler):
    def fit_single(self, data):
        return 1./data.abs().max().item()


class InterquartileScaler(BaseScaler):
    def fit_single(self, data):
        return 1. / float(RobustScaler(with_centering=False).fit(data.unsqueeze(1)).scale_)


class LipschitzScaler(BaseScaler):
    def __init__(self, likelihood, goal_smoothness, verbose=False):
        super(LipschitzScaler, self).__init__(likelihood, verbose)
        self.goal = float(goal_smoothness)

    def fit_single(self, dist, data, goal, index):
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

    def fit(self, data):
        def fit_recursive(dists, data, goal):
            if isinstance(dists, LikelihoodFlatten):
                old_value = dists.flatten
                dists.flatten = False

            pos = 0
            for d in dists:
                if isinstance(d, LikelihoodList) and not d.is_discrete:
                    num_dists = sum([len(x) for x in d if not x.is_discrete])
                    fit_recursive(d, data[..., pos: pos + d.domain_size], goal / num_dists)
                else:
                    for i in range(d.domain_size):
                        self.fit_single(d, data[..., pos + i], goal, index=i)
                pos += d.domain_size

            if isinstance(dists, LikelihoodFlatten):
                dists.flatten = old_value

        num_dists = sum([d.domain_size for d in self.likelihood if not d.is_discrete])
        fit_recursive(self.likelihood, data, self.goal / num_dists)
        return self.likelihood
