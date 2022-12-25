import torch
from torch.optim import Optimizer

class SGDHess(Optimizer):
   
    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        self.iteration = -1
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDHess, self).__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            for p in group['params']:
                state = self.state[p]
                state['prev_param'] = torch.zeros_like(p)
                state['current_param'] = torch.zeros_like(p)
                state['max_grad'] = torch.zeros_like(p)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.iteration += 1
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            vector = []
            grads = []
            param = []
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                prev_param, current_param, max_grad = state['prev_param'], state['current_param'], state['max_grad']
                with torch.no_grad():
                    if(self.iteration == 0):
                        prev_param.add_(p)
                        d_p = p.grad
                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)
                        if momentum != 0:
                            if 'momentum_buffer' not in state:
                                buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                            else:
                                buf = state['momentum_buffer']
                                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                            if nesterov:
                                d_p = d_p.add(buf, alpha=momentum)
                            else:
                                d_p = buf
                        max_grad.add_(d_p)
                        p.add_(d_p, alpha=-group['lr'])
                        current_param.copy_(p).detach()
                    else:
                        vector.append(current_param.add(prev_param, alpha = -1))
                        grads.append(p.grad)
                        param.append(p)
            if(self.iteration > 0):
                hvp = torch.autograd.grad(outputs = grads, inputs = param, grad_outputs=vector)
                with torch.no_grad():
                    i = 0
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        prev_param, current_param, max_grad = state['prev_param'], state['current_param'], state['max_grad'] 
                        with torch.no_grad():
                            prev_param.copy_(current_param).detach()
                            d_p = p.grad
                            if weight_decay != 0:
                                d_p = d_p.add(p, alpha=weight_decay)
                            if momentum != 0:
                                if 'momentum_buffer' not in state:
                                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                                else:
                                    buf = state['momentum_buffer']
                                    g = buf.add(hvp[i]).add(current_param-prev_param, alpha = weight_decay).mul(momentum).add(d_p, alpha=1 - dampening)
                                    #g = buf.add(hvp[i]).mul(1-momentum).add(d_p, alpha=momentum)
                                    val = None
                                    if(torch.norm(g) > torch.norm(max_grad)):
                                        val = max_grad*torch.div(g, torch.norm(g))
                                        max_grad.add_(g-max_grad)
                                    else:
                                        val = g
                                    buf.add_(val-buf)
                                    #buf.add_(hvp[i]).mul_(momentum).add_(d_p, alpha=1 - dampening)
                                if nesterov:
                                    d_p = d_p.add(buf, alpha=momentum)
                                else:
                                    d_p = buf
                            p.add_(d_p, alpha=-group['lr'])
                            current_param.copy_(p).detach()
                        i += 1
                            
        return loss