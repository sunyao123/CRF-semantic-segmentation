#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:59:11 2018

@author: sy
"""


import pydensecrf.densecrf as dense_crf
import numpy as np
import matplotlib.pyplot as plt
from densecrf2 import crf_model, potentials
from scipy.stats import multivariate_normal

# =============================================================================
# Create unary potential
# =============================================================================
#gt_prob: The certainty of the ground-truth (must be within (0,1)).
unary = potentials.UnaryPotentialFromProbabilities(gt_prob=0.7)

bilateral_pairwise = potentials.BilateralPotential(
    sdims=10,
    schan=0.01,
    compatibility=10,
    kernel=dense_crf.DIAG_KERNEL,
    normalization=dense_crf.NORMALIZE_SYMMETRIC
)

# =============================================================================
# Create CRF model and add potentials
# =============================================================================
#zero_unsure:  whether zero is a class, if its False, it means zero canb be any of other classes
crf = crf_model.DenseCRF(
    num_classes = 2,                  # The number of output classes
    zero_unsure = False,
    unary_potential=unary,
    pairwise_potentials=bilateral_pairwise,
    use_2d = 'non-rgb'                #'rgb-1d' or 'rgb-2d' or 'non-rgb'
)


# =============================================================================
# Load image and probabilities, use the original progect code
# =============================================================================
H, W, NLABELS = 400, 512, 2
# This creates a gaussian blob...
pos = np.stack(np.mgrid[0:H, 0:W], axis=2)
rv = multivariate_normal([H//2, W//2], (H//4)*(W//4))
probs = rv.pdf(pos)
# ...which we project into the range [0.4, 0.6]
probs = (probs-probs.min()) / (probs.max()-probs.min())
probs = 0.5 + 0.2 * (probs-0.5)

# The first dimension needs to be equal to the number of classes.
# Let's have one "foreground" and one "background" class.
# So replicate the gaussian blob but invert it to create the probability
# of the "background" class to be the opposite of "foreground".
probs = np.tile(probs[np.newaxis,:,:],(2,1,1))
probs[1,:,:] = 1 - probs[0,:,:]
probabilities = probs

NCHAN = 1
img = np.zeros((H,W,NCHAN), np.uint8)
img[H//3:2*H//3,W//4:3*W//4,:] = 1
image = img

# =============================================================================
# Set the CRF model
# =============================================================================
#label_source: whether label is from softmax, or other type of label.
crf.set_image(
    image=image,
    probabilities=probabilities,
    colour_axis=-1,                 # The axis corresponding to colour in the image numpy shape
    class_axis=0,                  # The axis corresponding to which class in the probabilities shape
    label_source = 'softmax'      #where the label come from, 'softmax' or 'label'
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








































