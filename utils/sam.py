import jittor as jt
from jittor import init
class SAM(jt.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.adaptive = adaptive

        self.base_optimizer = base_optimizer(params, **kwargs)
        self.params = list(params)
        self.state = {}

    def first_step(self):
        grad_norm = self._grad_norm()
        # print(f'[first_step] Total grad norm: {grad_norm.item():.6f}')
        scale = self.rho / (grad_norm + 1e-12)

        for i, p in enumerate(self.params):
            grad = p.opt_grad(self.base_optimizer)
            if grad is None:
                continue

            e_w = ((p ** 2) * grad * scale) if self.adaptive else (grad * scale)
            key = id(p)
            self.state[key] = {"old_p": p.clone()}
        
            if p.dtype.is_float():
                p.stop_grad()
            p[...] = p + e_w
            if p.dtype.is_float():
                p.start_grad()
          


    def second_step(self):
        for i, p in enumerate(self.params):
            if p.opt_grad(self.base_optimizer) is None:
                continue
            key = id(p)
            if key in self.state:
                # print("恢复前参数部分值:", p.flatten()[:5].tolist())
                if p.dtype.is_float():
                    p.stop_grad()
                p[...] = self.state[key]["old_p"]
                if p.dtype.is_float():
                    p.start_grad()
                # print("恢复后参数部分值:", p.flatten()[:5].tolist())

        self.base_optimizer.step()

    def _grad_norm(self):
        norm_list = []
        for i, p in enumerate(self.params):
            grad = p.opt_grad(self.base_optimizer)
            if grad is None:
                continue
            if self.adaptive:
                norm_val = (jt.abs(p) * grad).reshape(-1).norm(p=2)
                # print(
                #     f"[{i}] param shape: {list(p.shape)}, grad shape: {list(grad.shape)}, norm_val shape: {list(norm_val.shape)}")
            else:
                norm_val = grad.reshape(-1).norm(p=2)
            norm_list.append(norm_val)

        if len(norm_list) == 0:
            print("Warning: no gradients found for any parameters! Returning zero grad norm.")
            return jt.array(0.0)

        stacked = jt.stack(norm_list)
        # print(f"stacked norm_list shape: {list(stacked.shape)}")
        total_norm = jt.norm(stacked, p=2, dim=0)
        # print(f"total grad norm shape: {list(total_norm.shape)}")
        return total_norm
        # return jt.norm(jt.stack(norm_list), p=2)

    def zero_grad(self):
        for p in self.params:
            grad = p.opt_grad(self.base_optimizer)
            if grad is not None:
                jt.init.zero_(grad)
        self.base_optimizer.zero_grad()  # 如果 base_optimizer 有此方法，确保清零


    def load_state_dict(self, state_dict):
        self.state = state_dict.get('state', {})
        self.base_optimizer.load_state_dict(state_dict.get('base_optimizer', {}))

    def state_dict(self):
        return {
            'state': self.state,
            'base_optimizer': self.base_optimizer.state_dict()
        }