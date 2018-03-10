#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:04:33 2018

@author: sy
"""

import numpy as np
import pydensecrf.utils as crf_utils
import pydensecrf.densecrf as dense_crf


class Potential:
    pass


class UnaryPotential:
    pass


class PairwisePotential:
    pass


class UnaryPotentialFromProbabilities(UnaryPotential):
    def __init__(self, gt_prob):
        """Creates a unary potential from a probability mask.

        If the probabilities are for more than one class, then the first
        axis of the probabilitymask should specify which class it is a
        probability for. The shape of the probability mask for a 2D image
        should therefore be 
            [num_classes, height, width],
        the shape of 3D images should be
            [num_classes, height, width, depth],
        etc.

        In the case of only two classes, images with only one probability
        is accepted. I.e. the axis specifying which class the probabilities
        signify can be skipped.
        """
        self.__name__ = 'UnaryPotentialFromProbabilities'
        self.gt_prob = gt_prob  # The certainty of the ground-truth (must be within (0,1)).
    
    def apply(self, probability_mask, label_source, num_classes, zero_unsure, class_axis=0):
        if label_source =='label':
            if len(probability_mask.shape) == 2:
                anno_rgb = probability_mask.astype(np.uint32)
                _, labels = np.unique(anno_rgb, return_inverse=True)
                return crf_utils.unary_from_labels(labels, num_classes, gt_prob=self.gt_prob, zero_unsure=zero_unsure)                
            elif len(probability_mask.shape) == 3:
                anno_rgb = probability_mask.astype(np.uint32)
                anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
                
                # Convert the 32bit integer color to 1, 2, ... labels.
                # Note that all-black, i.e. the value 0 for background will stay 0.
                _, labels = np.unique(anno_lbl, return_inverse=True)
                return crf_utils.unary_from_labels(labels, num_classes, gt_prob=self.gt_prob, zero_unsure=zero_unsure)
            else:
                raise ValueError('The input shape must be 2 or 3')
        elif label_source == 'softmax':
            if class_axis != 0:    
                probability_mask = np.moveaxis(probability_mask, class_axis, 0)
            if probability_mask.shape[0] == 1:
                probability_mask = np.stack((probability_mask, 1-probability_mask),
                                        axis=0)
            return crf_utils.unary_from_softmax(probability_mask)




class AnisotropicBilateralPotential(PairwisePotential):
    def __init__(self, spatial_sigmas, colour_sigmas, compatibility):
        r"""Creates a bilateral potential for image segmentation with DenseCRF.

        The bilateral energy-function is on the form:
        .. math::
           \mu exp(-(x_i-x_j)^T\Sigma_x^{-1}(x_i-x_j)
                   -(c_i-c_j)^T\Sigma_c^{-1}(c_i-c_j)),
        
        where :math:`x_i` and :math:`c_i` is respectively the position
        and colour of pixel i. :math:`\mu` is the (inverse) compatibility
        between the label of pixel i and pixel j. The :math:`\Sigma` matrices
        are created by `diag(spatial_sigmas)` and `diag(colour_sigma)`
        respectively.

        Arguments
        ---------
        spatial_sigmas : numpy.ndarray
            Specifies how fast the energy should decline with spatial distance.
        colour_sigmas : numpy.ndarray
            Specifies how fast the energy should decline with colour distance.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.__name__ = 'AnisotropicBilateralPotential'
        self.spatial_sigmas = spatial_sigmas
        self.colour_sigmas = colour_sigmas
        self.compatibility = compatibility
    
    def apply(self, image, colour_axis):
        return crf_utils.create_pairwise_bilateral(
            img=image,
            sdims=self.spatial_sigmas,
            schan=self.colour_sigmas,
            chdim=colour_axis
        )


class BilateralPotential(PairwisePotential):
    def __init__(self, sdims, schan, compatibility, kernel=dense_crf.DIAG_KERNEL,
                          normalization=dense_crf.NORMALIZE_SYMMETRIC):
        r"""Creates a bilateral potential for image segmentation with DenseCRF.

        The bilateral energy-function is on the form:
        .. math::
           \mu exp(-\frac{||x_i-x_j||^2}{\sigma_x}
                   -\frac{||c_i-c_j||^2}{\sigma_c}),
        
        where :math:`x_i` and :math:`c_i` is respectively the position
        and colour of pixel i. :math:`\mu` is the (inverse) compatibility
        between the label of pixel i and pixel j.

        Arguments
        ---------
        spatial_sigma : float
            Specifies how fast the energy should decline with spatial distance.
        colour_sigma : float
            Specifies how fast the energy should decline with colour distance.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.__name__ = 'BilateralPotential'
        self.spatial_sigma = sdims
        self.colour_sigma = schan
        self.compatibility = compatibility
        
        self.kernel = kernel
        self.normalization = normalization
        
    def apply(self, image, colour_axis):
        spatial_sigmas = [self.spatial_sigma for _ in range(len(image.shape)-1)]
        colour_sigmas = [self.colour_sigma 
                            for _ in range(image.shape[colour_axis])]
        return crf_utils.create_pairwise_bilateral(
            sdims=spatial_sigmas,
            schan=colour_sigmas,
            img=image,
            chdim=colour_axis
        )


class AnisotropicGaussianPotential(PairwisePotential):
    def __init__(self, sigmas, compatibility):
        r"""Creates an anisotropic Gaussian potential for image segmentation.

        The anisotropic Gaussian energy-function is on the form:
        .. math::
           \mu exp(-(x_i-x_j)^T\Sigma^{-1}(x_i-x_j)),
        
        where :math:`x_i` is the position pixel i. :math:`\mu` is the 
        (inverse) compatibility between the label of pixel i and pixel j. The
        :math:`\Sigma` matrix is given by diag(`sigmas`).

        Arguments
        ---------
        sigmas : numpy.ndarray
            Specifies how fast the energy should decline with spatial distance
            along each of the axes.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.__name__ = 'AnisotropicGaussianPotential'
        self.sigmas = sigmas
        self.compatibility = compatibility
    
    def apply(self, image, colour_axis):
        if colour_axis == -1:
            colour_axis = len(image.shape) - 1
        shape = [
            image.shape[i] for i in range(len(image.shape)) if i != colour_axis
        ]
        return crf_utils.create_pairwise_gaussian(
            sdims=self.sigmas,
            shape=shape
        )


class GaussianPotential(PairwisePotential):
    def __init__(self, sigma, compatibility, kernel=dense_crf.DIAG_KERNEL,
                          normalization=dense_crf.NORMALIZE_SYMMETRIC):
        r"""Creates a Gaussian potential for image segmentation with DenseCRF.

        The Gaussian energy-function is on the form:
        .. math::
           \mu exp(-\frac{||x_i-x_j||^2}{\sigma}}),
        
        where :math:`x_i` is the position pixel i. :math:`\mu` is the 
        (inverse) compatibility between the label of pixel i and pixel j.

        Arguments
        ---------
        sigma : float
            Specifies how fast the energy should decline with spatial distance.
        compatibility : float or numpy.ndarray
            The (inverse) compatibility function. If constant, a Potts-like
            potential is used, if it is a matrix, then the element
            compatibility[i, j] specifies the cost of having label i adjacent
            to a pixel with label j. High `compatibility` values yields a
            strong potential.
        """
        self.__name__ = 'GaussianPotential'
        self.sigma = sigma
        self.compatibility = compatibility
        self.kernel = kernel
        self.normalization = normalization
    
    def apply(self, image, colour_axis):
        spatial_sigmas = [self.sigma for _ in range(len(image.shape)-1)]
        if colour_axis == -1:
            colour_axis = len(image.shape) - 1
        shape = [image.shape[i] for i in range(len(image.shape)) if i != colour_axis]

        return crf_utils.create_pairwise_gaussian(
            sdims=spatial_sigmas, shape=shape
        )
