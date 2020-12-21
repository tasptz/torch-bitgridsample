## What does it do?
It works like torch `grid_sample` with bilinear interpolation and zero padding, but takes in uint8 datatype with shape (B, H, W), where
each bit of a pixel corresponds to a channel. Encoding can be done like
```
sum(x[:, i] * 2**i for i in range(x.shape[1]))
```
## Why?
To safe GPU memory. Consider the case where you want to sample small regions from large images, which only consist of binary masks. E.g. CMYK printing motives. Torch `grid_sample` only works for floating point types, so in
case of 4 channels, memory consumption is reduced by `sizeof(float) * channels = 4 * 4 = 16`.
## Installation
Edit the arguments `BLOCKSIZE` in `setup.py`, accordingly to your requirements.
Then execute `setup.py` as needed, e.g.
```
python setup.py develop
```
