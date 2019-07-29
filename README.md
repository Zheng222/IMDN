# IMDN
Lightweight Image Super-Resolution with Information Multi-distillation Network (ACM MM 2019)

## PSNR & SSIM
<p align="center">
    <img src="images/psnr_ssim.png" width="960"> <br />
    <em> Average PSNR/SSIM on datasets Set5, Set14, BSD100, Urban100, and Manga109. </em>
</p>

## Model parameters

<p align="center">
    <img src="images/parameters.png" width="480"> <br />
    <em> Trade-off between performance and number of parameters on Set5 ×4 dataset. </em>
</p>

## Running time

<p align="center">
    <img src="images/time.png" width="480"> <br />
    <em> Trade-off between performance and running time on Set5 ×4 dataset. VDSR, DRCN, and LapSRN were implemented by MatConvNet, while DRRN, and IDN employed Caffe package. The rest EDSR-baseline, CARN, and our IMDN utilized PyTorch. </em>
</p>
