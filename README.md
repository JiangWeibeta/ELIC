# ELIC
Unofficial Implementation of CVPR22 papar "ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding".

My implementation is based on CompressAI.

**Code will be available soon :)**

I trained a model model optimized for MSE. $\lambda$ is set to $0.08$. The performance of the ELIC reproduced by me has the same rate-distortion with the official ELIC on Kodak and better performance than official ELIC on CLIC 2021 Test dataset.

<p float="left">
  <img src="https://github.com/JiangWeibeta/ELIC/blob/main/results/elic_reproduce_psnr.png" width="400" />
  <img src="https://github.com/JiangWeibeta/ELIC/blob/main/results/elic_reproduce_psnr_clic.png" width="400" />
</p>
