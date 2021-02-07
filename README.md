# lipstd: Lipschiz standardization on Pytorch

This is an open-source implementation of [Lipschitz standardization](https://arxiv.org/abs/2002.11369) for probabilistic 
machine learning in Pytorch.

The library introduces different mechanisms to incorporate the likelihood declaration into existing code in a way as 
transparent as possible. That is, it enables the user to declare the likelihood of the data, and carry that information
throughout the entire training pipeline. This includes:
- Likelihood declaration
  ```python
  likelihood = get_likelihood('gamma', 'normal')
  ```
- Likelihood instantiation for specific parameters (which returns usual `torch.distribution` classes). By default, the
  likelihoods will automatically make sure that the parameters sent fulfill the distributional constraints (for example,
  if they need to be positive they will be passed through a softplus function in the background).
  ```python
  p_x = likelihood(gamma_eta1, gamma_eta2, normal_eta1, normal_eta2)  # Distributions are parameterized by the natural parameters
  
  # You can call usual Pytorch functions
  data = p_x.sample([1000])  
  print(p_x.log_prob(data).mean(dim=0).tolist()) 
  ```
- Preprocessing the data using Lipschitz standardization (`LipschitzScaler` function)
  ```python
  scaler = LipschitzScaler(likelihood, 1 / 0.001)
  scaler.fit(data)
  ```
- Include data scaling into the model pipeline (using the attribute `InputScaler` of any likelihood before passsing the 
  data to the model)
  ```python
  net = nn.Sequential(
    likelihood.InputScaler,  # This will scale the data on the fly
    nn.Linear(likelihood.domain_size, 100), nn.Tanh(),  # domain_size = number of dimensions of the likelihood
    nn.Linear(100, likelihood.num_params)  # num_params = number of parameters of the likelihood
  )
  p_x = likelihood(*net(x))
  ```  
- Automatic evaluation on the original likelihood and the scaled likelihood during training and test time, respectively. 
  This is done transparently and switching between modes through the usual Pytorch calls `.train()` and `.eval()`. By 
  storing the likelihood as an attribute of your model this should done without further changes. To be more clear:
  - During training data is scaled before being evaluated.
  - During evaluation parameters are scaled back to the original space before instantiation.

## Implemented likelihoods
- Continuous distributions: Normal/Gaussian, Log-normal, Gamma, Exponential
- Discrete distributions: Categorical, One-Hot Categorical, Poisson, Bernoulli
- Dequantization techniques: GammaTrick (as in [the original paper](https://arxiv.org/abs/2002.11369))