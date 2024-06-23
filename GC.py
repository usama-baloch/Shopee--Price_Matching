import torch
import math
from torch.optim.optimizer import Optimizer

#credit : https://github.com/Yonghongwei/Gradient-Centralization

"""
according to the tests performed by the author of Ranger Optimizer, The GC + Ranger
optimizer will perform too well and if you specifically use Mish activation then it
will even work better, so try replacing the model's activation with mish and 
check the results.

"""

def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    
    """
    use_gc=True means algorithm add GC operation otherwise not.
    gc_conv_only=True means algorithm add GC operation to the 
    Conv Layer otherwise both Conv and FC Layer.

    """
        
    if use_gc:
        if gc_conv_only:
            
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


"""
    Ranger is a Combination of Two Newly developed Optimizers (RAdam + Lookahead)
    it is very efficient and stable and widely used in deep neural network
"""
class Ranger(Optimizer):
    
    def __init__(self, params, lr=1e-3, 
                alpha = 0.5, k=5, N_sma_threshhold=5,
                betas=(0.95, 0.999), eps=1e-5, weight_decay=0,
                
                use_gc=True, gc_conv_only=False, gc_loc=True):
        
        
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')
            
        
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                       N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        
        super().__init__(params, defaults)
        
        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold
        self.alpha = alpha
        self.k = k
        
        # radam buffer for state 
        
        self.radam_buffer = [[None, None, None] for _ in range(10)]
        
        # gc on or off
        self.use_gc = use_gc
        self.gc_conv_only=gc_conv_only
        self.gc_loc = gc_loc
        
        
        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")
            
        
    def __setstate__(self, state):
        print('set state called')
        super(Ranger, self).__setstate__(state)
    
    def step(self, clouser=None):
        
        loss = None
        
        
        for group in self.param_groups:
            
            for parameter in group['params']:
                
                if parameter is None:
                    continue
                grad = parameter.grad.data.float()
                
                if grad.is_sparse:
                    raise RuntimeError("Ranger Optimizer does not support sparse gradient ")
                
                parameter_data_fp32 = parameter.data.float()
                state = self.state[parameter]
                
                # if first time to run...init dictionary with our desired entries
                if len(state) == 0:
                    
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(parameter_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(parameter_data_fp32)
                    
                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(parameter.data)
                    state['slow_buffer'].copy_(parameter.data)
                    
                else:
                    
                    state['exp_avg'] = state['exp_avg'].type_as(parameter_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(parameter_data_fp32)
                
                # start our computation
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc = self.use_gc, 
                                                gc_conv_only = self.gc_conv_only)
                
                state['step'] += 1
                
                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                # compute mean mov avg
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                
                buffered = self.radam_buffer[int(state['step'] % 10)]
                
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[0], buffered[1]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    
                    buffered[1] = N_sma
                    
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                    
                    
            # apply lr
            if N_sma > self.N_sma_threshhold:
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                G_grad = exp_avg / denom
            else:
                G_grad = exp_avg
            
            if group['weight_decay'] != 0:
                G_grad.add_(parameter_data_fp32, alpha=group['weight_decay'])
            
            # GC operation
            if self.gc_loc == False:
                G_grad = centralized_gradient(G_grad, use_gc = self.use_gc, 
                                              gc_conv_only = self.gc_conv_only)
            parameter_data_fp32.add_(G_grad, alpha = -step_size * group['lr'])
            parameter.data.copy_(parameter_data_fp32)
            
            
            if state['step'] % group['k'] == 0:
                
                # get access to slow parameter tensor
                slow_p = state['slow_buffer']
                # (fast weights - slow weights) * alpha
                slow_p.add_(parameter.data - slow_p, alpha=self.alpha)
                # copy interpolated weights to RAdam param tensor
                parameter.data.copy_(slow_p)
        
        return loss