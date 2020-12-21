import torch
import torch.nn.functional as F
from torch.autograd.gradcheck import get_numerical_jacobian
# torch.manual_seed(0)
import numpy as np

from bitgridsample import bit_grid_sample

def _encode(x):
    return sum(x[:, i] * 2**i for i in range(x.shape[1]))

def _plot(fn, a, b):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2)
    def _f(x):
        return x.detach().cpu().numpy()
    ax[0].imshow(_f(a))
    ax[1].imshow(_f(b))
    fig.savefig(f'{fn}.pdf', dpi=300)
    plt.close(fig)

def _R2(angle):
    B = len(angle)
    c = torch.cos(angle)
    s = torch.sin(angle)
    return torch.stack((c, -s, s, c), dim=1).view(B, 2, 2)

def _bit_grid_sample(img_in_encoded, grid, channels=3):
    x = img_in_encoded.cpu().numpy()
    x = np.stack([np.bitwise_and(np.right_shift(x, i), 1) for i in range(channels)], axis=1)
    x = grid.new_tensor(x)
    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return y

def test_bitgridsample():
    device = torch.device('cuda:0')
    B, C, W, H = 4, 3, 32, 16
    img_in = torch.empty((B, C, H, W), dtype=torch.uint8, device=device).random_(0, 2)
    img_in_encoded = _encode(img_in)

    M = _R2(torch.empty(B, dtype=torch.float32, device=device).uniform_(-np.pi, np.pi))
    M *= M.new_empty((B, 1, 1)).uniform_(0.5, 1.5)
    M = torch.cat((M, M.new_empty((B, 2, 1)).uniform_(-0.2, 0.2)), dim=2)

    shape_out = 16, 32
    grid = F.affine_grid(M, (B, C) + shape_out, align_corners=False)
    grid.requires_grad_(True)
    def f(g):
        grad_test = grid.new_zeros((B, C,) + shape_out + (2,))
        for b in range(B):
            for r in range(shape_out[0]):
                for c in range(shape_out[1]):
                    for ch in range(C):
                        img_out = g(
                            img_in_encoded,
                            grid,
                            channels=C
                        )
                        img_out[b, ch, r, c].backward()
                        grad_test[b, ch, r, c] = grid.grad.data[b, r, c].clone()
                        grid.grad.data.zero_()
        return grad_test, img_out

    grad_test, img_out_test = f(_bit_grid_sample)
    grad, img_out = f(bit_grid_sample)

    # for b in range(B):
    #     for ch in range(C):
    #         _plot(f'{b}_{ch}_out', img_out[b, ch], img_out_test[b, ch])
    #         for i, n in enumerate((f'{b}_{ch}_gx', f'{b}_{ch}_gy')):
    #             _plot(n, grad[b, ch, :, :, i], grad_test[b, ch, :, :, i])

    assert torch.allclose(img_out.contiguous(), img_out_test, atol=1e-3, rtol=0)
    assert torch.allclose(grad, grad_test, atol=1e-3, rtol=0)

if __name__ == '__main__':
    test_bitgridsample()