# Token Layout Proof

PyTorch source layout is contiguous `[B,T,H,W,C]`. Fixed window layout is `[num_windows,tokens,C]`.
For the fixture, windows enumerate `(b, block_t, block_h, block_w)` and tokens enumerate `(local_t, local_h, local_w)` with `local_w` varying fastest.

## First window

| token index | (t,h,w) | channel-0 value |
|---:|---|---:|
| 0 | (0,0,0) | 0 |
| 1 | (0,0,1) | 1 |
| 2 | (0,0,2) | 2 |
| 3 | (0,0,3) | 3 |
| 4 | (0,1,0) | 10 |
| 5 | (0,1,1) | 11 |
| 6 | (0,1,2) | 12 |
| 7 | (0,1,3) | 13 |
| 8 | (0,2,0) | 20 |
| 9 | (0,2,1) | 21 |
| 10 | (0,2,2) | 22 |
| 11 | (0,2,3) | 23 |
| 12 | (0,3,0) | 30 |
| 13 | (0,3,1) | 31 |
| 14 | (0,3,2) | 32 |
| 15 | (0,3,3) | 33 |
| 16 | (1,0,0) | 100 |
| 17 | (1,0,1) | 101 |
| 18 | (1,0,2) | 102 |
| 19 | (1,0,3) | 103 |
| 20 | (1,1,0) | 110 |
| 21 | (1,1,1) | 111 |
| 22 | (1,1,2) | 112 |
| 23 | (1,1,3) | 113 |
| 24 | (1,2,0) | 120 |
| 25 | (1,2,1) | 121 |
| 26 | (1,2,2) | 122 |
| 27 | (1,2,3) | 123 |
| 28 | (1,3,0) | 130 |
| 29 | (1,3,1) | 131 |
| 30 | (1,3,2) | 132 |
| 31 | (1,3,3) | 133 |

## ncnn mapping

ncnn::Mat does not preserve a PyTorch five-dimensional layout directly. The C++ runner stores a single fixture tensor as `w=C`, `h=W`, `d=H`, `c=T` (batch is fixed at 1). Thus element `(t,h,w,c)` is addressed as `mat.channel(t).depth(h).row(w)[c]`. A window is represented as `w=C`, `h=tokens`, `c=num_windows`; element `(window,token,channel)` is `mat.channel(window).row(token)[channel]`.
