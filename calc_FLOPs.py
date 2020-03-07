from model import architecture
from FLOPs.profile import profile

width = 360
height = 240
model = architecture.IMDN_RTC(upscale=2)
flops, params = profile(model, input_size=(1, 3, height, width))
print('IMDN_light: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format(height,width,flops/(1e9),params))
