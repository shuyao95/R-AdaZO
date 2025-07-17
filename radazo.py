import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class R_AdaZO(Optimizer):
    """
    Implements R-AdaZO algorithm (Refining Adaptive Zeroth-Order Optimization).
    
    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        mu (float, optional): smoothing parameter for gradient estimation (default: 1e-3)
        n_samples (int, optional): number of samples for gradient estimation (default: 2)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, mu=5e-3, n_samples=2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= mu:
            raise ValueError(f"Invalid mu parameter: {mu}")
        if not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError(f"Invalid n_samples parameter: {n_samples}")

        defaults = dict(lr=lr, betas=betas, eps=eps, mu=mu, n_samples=n_samples)
        super(R_AdaZO, self).__init__(params, defaults)
        self.state['seeds'] = []
        self.state['step'] = 0

    def _reset_seeds(self, new_seeds=None):
        if new_seeds is None:
            new_seeds = torch.randint(
                0, 2**32 - 1, (self.defaults['n_samples'],), dtype=torch.int64
            ).tolist()
        self.state['seeds'] = new_seeds

    def _get_perturbation(self, shape, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        u = torch.randn(shape, generator=generator)
        u_norm = torch.norm(u)
        if u_norm > 0:
            u = u / u_norm
        return u

    @torch.no_grad()
    def _estimate_gradient(self, p, closure, mu, seed):
        shape = p.shape
        u = self._get_perturbation(shape, seed)
        
        # Evaluate f(x + mu * u)
        p.data.add_(mu * u)
        f_plus = closure()
        
        # Evaluate f(x)
        p.data.add_(- mu * u)
        f = closure()
        
        # Compute gradient estimate
        grad_est = (f_plus - f) * u / mu
        
        return grad_est, f

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError("RaZO requires closure to evaluate function value")

        self.state['step'] += 1
        for group in self.param_groups:
            for p in group['params']:

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                mu = group['mu']
                n_samples = group['n_samples']
                beta1, beta2 = group['betas']
                
                grad_est = torch.zeros_like(p)
                for seed in self.state['seeds']:
                    current_grad, loss = self._estimate_gradient(p, closure, mu, seed)
                    grad_est.add_(current_grad)

                grad_est.div_(n_samples)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                exp_avg.mul_(beta1).add_(grad_est, alpha=1 - beta1)
                # Only one line change to standard ZO-AdaMM, which is the key of R-AdaZO
                exp_avg_sq.mul_(beta2).addcmul_(exp_avg, exp_avg, value=(1 - beta2))
                
                step_size = group['lr']
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
