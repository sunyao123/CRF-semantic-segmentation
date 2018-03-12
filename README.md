# CRF-semantic-segmentation<br>
使用CRF进行语义分割<br>
[论文链接](https://arxiv.org/abs/1210.5644)<br>
[pydensecrf模块](https://github.com/lucasb-eyer/pydensecrf)<br>

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
主要有两个文件crf_model.py和potentials.py<br>
crf_model.py建立CRF模型；potentials.py生成所需要的势能。<br>
图像可以输入RGB也可以输入灰度图像。<br>
RGB可以使用1d或者2d进行分割。<br>

具体例子看examples<br>


原始pydensecrf库：<br>
对于添加单点势能，DenseCRF()和DenseCRF2D()的方法是一样的。<br>
如果使用DenseCRF()函数创建模型，则在添加pairwise能量时必须先使用create_pairwise_gaussian()或者create_pairwise_bilateral()来生成，再通过addPairwiseEnergy()函数添加。<br>
如果使用DenseCRF2D()函数创建模型，则可以直接通过addPairwiseGaussian()和addPairwiseBilateral()来添加；而且addPairwiseBilateral()必须要求label是3通道(有颜色，公式里有)。<br>

