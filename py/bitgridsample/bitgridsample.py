import torch

from . import bitgridsample_cuda

class BitGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid, channels):
        with torch.no_grad():
            assert str(x.device)[:4]  == 'cuda'
            assert str(grid.device)[:4]  == 'cuda'
            assert sum(ctx.needs_input_grad) == int(ctx.needs_input_grad[1])
            shape = grid.shape[:3]
            out = grid.new_empty(shape + (channels,)).contiguous()
            out_grad = grid.new_empty(shape + (channels, 2)).contiguous()
            bitgridsample_cuda.forward(
                x.contiguous(),
                grid.squeeze().contiguous(),
                out,
                out_grad
            )      
            ctx.save_for_backward(out_grad)
            return out.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1]:
            H, W = grad_output.shape[2:]
            g_grid = ctx.saved_tensors[0]
            return None, (grad_output[..., None] * g_grid.permute(0, 3, 1, 2, 4)).sum(axis=1), None
        return None, None, None

def bit_grid_sample(x, grid, channels=3):
    return BitGridSample.apply(x, grid, channels)