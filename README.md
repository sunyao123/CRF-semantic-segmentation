# CRF-semantic-segmentation<br>
使用CRF进行语义分割<br>
论文链接：<br>
https://arxiv.org/abs/1210.5644<br>
python模块<br>
https://github.com/lucasb-eyer/pydensecrf<br>

本模块调用pydensecrf，进行封装<br>
彩图：可以使用1d，也可以使用2d。<br>
灰度图：使用2d。因为原始模块pydensecrf的2d模型里边函数addPairwiseBilateral()要求原图为rgb。<br>

可以使用两种方式运行inference。<br>
1：直接跑完指定循环;<br>
```
crf.perform_inference(n)
mask = crf.segmentation_map
plt.imshow(mask)
plt.show()
```

2：观察每次KL_divergence;<br>
```
crf.perform_step_inference(n)
mask = crf.segmentation_map
plt.imshow(mask)
plt.show()
```

具体例子看examples<br>


