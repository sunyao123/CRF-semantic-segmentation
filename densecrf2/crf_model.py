#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:04:33 2018

@author: sy
"""



import numpy as np
import pydensecrf.densecrf as dense_crf
#from . import potentials
from densecrf2 import potentials

class DenseCRF:
    def __init__(self, num_classes, zero_unsure, unary_potential, pairwise_potentials, use_2d = 'rgb-1d'):
        """A wrapper for the PyDenseCRF functions. 
        
        Currently only 2D images are supported.
        
        Arguments
        ---------
        num_classes : int
        unary_potential : UnaryPotential
        pairwise_potentials : PairwisePotential or array like
            A collection of PairwisePotential instances.
        """
        if isinstance(pairwise_potentials, potentials.PairwisePotential):
            pairwise_potentials = [pairwise_potentials]

        for pairwise_potential in pairwise_potentials:
            self.check_potential(pairwise_potential, potentials.PairwisePotential)
        self.check_potential(unary_potential, potentials.UnaryPotential)

        self.zero_unsure = zero_unsure
        self.pairwise_potentials = pairwise_potentials
        self.unary_potential = unary_potential
        
        self.use_2d = use_2d
        self.num_classes = num_classes
        self.crf_model = None
        self._image = None
        self._colour_axis = None

        # Variables needed for the CRF model to run
        self._Q = None
        self._tmp1 = None
        self._tmp2 = None

    def set_image(self, image, probabilities, colour_axis=None, class_axis=None, label_source='label'):
        """Set the image for the CRF model to perform inference on.

        Arguments
        ---------
        image : numpy.ndarray
            The image to segment.
        probabilities : numpy.ndarray
            Class probabilities for each pixel.
        colour_axis : int
            Which axis of the image array the colour information lies in.
            Usually the last axis (-1), but can also be the first axis (0).
            If `image` is a grayscale image, this should be set to None.
        class_axis : int
            Which axis of the probabilities array the class information lies.
            If the probabilites are on the form
                ``[image_height, image_width, class]``,
            then class axis should be set to either `3` or `-1`.
        """
        if colour_axis is None:
            colour_axis = -1
            image = image.reshape(*image.shape, 1)

        self._image = image
        self._colour_axis = self.fix_negative_index(colour_axis)
        self._class_axis = class_axis
        self._probabilities = probabilities
        self._image_shape = \
            image.shape[:self._colour_axis] + image.shape[self._colour_axis+1:]
        self.label_source = label_source

##        'rgb-1d' or 'rgb-2d' or 'non-rgb'
        if self.use_2d == 'rgb-1d':
            self.process_rgb_label()
            self._create_model()
            self._set_potentials_1d()
        elif self.use_2d == 'rgb-2d':
            self.process_rgb_label()
            self._create_2d_model()
            self._set_potentials_2d()
        elif self.use_2d == 'non-rgb':
            self._create_2d_model()
            self._set_potentials_non_rgb()
        else:
            raise ValueError('There is no {0} format. Please input one of the rgb-1d, rgb-2d or non-rgb'.format(self.use_2d))
            
    def _create_model(self):
        self.crf_model = dense_crf.DenseCRF(np.prod(self.image_shape),
                                            self.num_classes)

    def _create_2d_model(self):
        self.crf_model = dense_crf.DenseCRF2D(self.image_shape[1], self.image_shape[0],
                                            self.num_classes)


    def _set_potentials_1d(self):
        """Apply the potentials to current image.
        """
        self.crf_model.setUnaryEnergy(
            self.unary_potential.apply(self._probabilities, self.label_source, self.num_classes, self.zero_unsure, self._class_axis)
        )

        for pairwise_potential in self.pairwise_potentials:
            self.crf_model.addPairwiseEnergy(
                pairwise_potential.apply(self._image, self._colour_axis),
                compat=pairwise_potential.compatibility,
                kernel=dense_crf.DIAG_KERNEL,
                normalization=dense_crf.NORMALIZE_SYMMETRIC
            )

    def _set_potentials_2d(self):
        """Apply the potentials to current image.
        """
        self.crf_model.setUnaryEnergy(
            self.unary_potential.apply(self._probabilities, self.label_source, self.num_classes, self.zero_unsure, self._class_axis)
        )

        for pairwise_potential in self.pairwise_potentials:
            
            if isinstance(pairwise_potential, potentials.GaussianPotential):
                    # This adds the color-independent term, features are the locations only.
                self.crf_model.addPairwiseGaussian(sxy=pairwise_potential.sigma, compat=pairwise_potential.compatibility, kernel=pairwise_potential.kernel,
                                      normalization=pairwise_potential.normalization)
            
            if isinstance(pairwise_potential, potentials.BilateralPotential):
                # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
                self.crf_model.addPairwiseBilateral(sxy=pairwise_potential.spatial_sigma, srgb=pairwise_potential.colour_sigma, rgbim=self._image,
                                       compat=pairwise_potential.compatibility,
                                       kernel=pairwise_potential.kernel,
                                       normalization=pairwise_potential.normalization)

    def _set_potentials_non_rgb(self):
        """Apply the potentials to current image.
        """
        self.crf_model.setUnaryEnergy(
            self.unary_potential.apply(self._probabilities, self.label_source, self.num_classes, self.zero_unsure, self._class_axis)
        )

        for pairwise_potential in self.pairwise_potentials:
            self.crf_model.addPairwiseEnergy(
                pairwise_potential.apply(self._image, self._colour_axis),
                compat=pairwise_potential.compatibility,
                kernel=dense_crf.DIAG_KERNEL,
                normalization=dense_crf.NORMALIZE_SYMMETRIC)


    def _start_inference(self):
        """Prepare the model for inference."""
        self._Q, self._tmp1, self._tmp2 = self.crf_model.startInference()

    def perform_step_inference(self, num_steps):
        """Perform `num_steps` iterations in the CRF energy minimsation.

        The minimisation continues where it previously left off. So calling
        this function twice with `num_steps=10` is the same as calling it
        once with `num_steps=20`.
        """
        self._start_inference()
        for i in range(num_steps):
            print("KL-divergence at {}: {}".format(i, self.kl_divergence))
            self.crf_model.stepInference(self._Q, self._tmp1, self._tmp2)

    def perform_inference(self, num_steps):
        """Perform `num_steps` iterations in the CRF energy minimsation.

        The minimisation continues where it previously left off. So calling
        this function twice with `num_steps=10` is the same as calling it
        once with `num_steps=20`.
        """
        self._Q = self.crf_model.inference(num_steps)

    def fix_negative_index(self, idx):
        if idx < 0:
            idx = len(self._image.shape) + idx
        return idx

    def process_rgb_label(self):
        anno_rgb = self._probabilities.astype(np.uint32)
        anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
        # Convert the 32bit integer color to 1, 2, ... labels.
        # Note that all-black, i.e. the value 0 for background will stay 0.
        colors, labels = np.unique(anno_lbl, return_inverse=True)
        # But remove the all-0 black, that won't exist in the MAP!
        HAS_UNK = 0 in colors
        if HAS_UNK:
            if self.zero_unsure:
                self.colors = colors[1:]
                self.num_classes = len(set(labels.flat)) - 1
            else:
                self.num_classes = len(set(labels.flat))
        else:
            self.num_classes = len(set(labels.flat))
        #else:
        #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")
        
        # And create a mapping back from the labels to 32bit integer colors.
        colorize = np.empty((len(colors), 3), np.uint8)
        colorize[:,0] = (colors & 0x0000FF)
        colorize[:,1] = (colors & 0x00FF00) >> 8
        colorize[:,2] = (colors & 0xFF0000) >> 16
        self.colorize = colorize

    @staticmethod
    def check_potential(potential, potential_type=None):
        """Checks `potential` is of correct type and has an apply function.
        """
        potential_type = potentials.Potential if potential_type is None \
                                              else potential_type
        potential_name = potential_type.__name__
        if not isinstance(potential, potential_type):
            raise ValueError(
                '{0} is not a {1}'.format(potential.__name__, potential_name)
            )
        elif not hasattr(potential, 'apply'):
            raise ValueError(
                '{0} is not yet implemented.'.format(potential.__name__)
            )

    @property
    def kl_divergence(self):
        """The KL divergence"""
        if self._Q is None:
            raise RuntimeWarning('No image is set')
            return None
        return self.crf_model.klDivergence(self._Q)/np.prod(self.image_shape)
    
    @property
    def segmentation_map(self):
        if self._Q is None:
            raise RuntimeWarning('No image is set')
            return None
        if self.use_2d == 'non-rgb':
            return np.argmax(self._Q, axis=0).reshape(self.image_shape)
        else:
            MAP = np.argmax(self._Q, axis=0)
            MAP = self.colorize[MAP,:]
            return MAP.reshape(self._image.shape)
        
#    @property
#    def image(self):
#        return self._image.copy()
    
    @property
    def image_shape(self):
        return self._image_shape
    
  