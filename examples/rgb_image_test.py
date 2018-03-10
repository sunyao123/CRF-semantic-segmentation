#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:10:58 2017

@author: sy
"""


import pydensecrf.densecrf as dense_crf
from cv2 import imread
import matplotlib.pyplot as plt
from densecrf2 import crf_model, potentials

# Create unary potential
unary = potentials.UnaryPotentialFromProbabilities(gt_prob=0.7)

bilateral_pairwise = potentials.BilateralPotential(
    sdims=80,
    schan=13,
    compatibility=4,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

gaussian_pairwise = potentials.GaussianPotential(
    sigma=3, 
    compatibility=2,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

# =============================================================================
# Create CRF model and add potentials
# =============================================================================
#zero_unsure:  whether zero is a class, if its False, it means zero canb be any of other classes
# =============================================================================
# crf = crf_model.DenseCRF(
#     num_classes = 3,
#     zero_unsure = True,              # The number of output classes
#     unary_potential=unary,
#     pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
#     use_2d = 'rgb-2d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
# )
# =============================================================================
crf = crf_model.DenseCRF(
    num_classes = 3,
    zero_unsure = True,              # The number of output classes
    unary_potential=unary,
    pairwise_potentials=[bilateral_pairwise, gaussian_pairwise],
    use_2d = 'rgb-1d'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
)


# =============================================================================
# Load image and probabilities
# =============================================================================
image = imread('./im.png')
probabilities = imread('./label.png')

# =============================================================================
# Set the CRF model
# =============================================================================
#label_source: whether label is from softmax, or other type of label.
crf.set_image(
    image=image,
    probabilities=probabilities,
    colour_axis=-1,                  # The axis corresponding to colour in the image numpy shape
    class_axis=-1,                   # The axis corresponding to which class in the probabilities shape
    label_source = 'label'           # where the label come from, 'softmax' or 'label'
)

# =============================================================================
# run the inference
# =============================================================================
# Run 10 inference steps.
crf.perform_step_inference(10)
mask0 = crf.segmentation_map

# Run 80 inference steps.
crf.perform_step_inference(80)  # The CRF model will restart run.
mask80 = crf.segmentation_map

# Plot the results
plt.subplot(121)
plt.title('Segmentation mask after 10 iterations')
plt.imshow(mask0)

plt.subplot(122)
plt.title('Segmentation mask after 80 iterations')
plt.imshow(mask80)
plt.show()


crf.perform_inference(80)  # The CRF model will restart run.
new_mask80 = crf.segmentation_map
print(crf.kl_divergence)
plt.imshow(new_mask80)
plt.show()












