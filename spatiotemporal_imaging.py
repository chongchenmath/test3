#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:40:32 2018

@author: chench

Spatiotemporal image reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
from odl.operator import (Operator, IdentityOperator, ScalingOperator,
                          ConstantOperator, DiagonalOperator, PointwiseNorm,
                          MultiplyOperator)
from odl.space import ProductSpace
from odl.set.space import LinearSpaceElement
import matplotlib.pyplot as plt
from odl.tomo import Parallel2dGeometry, RayTransform, fbp_op
from odl.solvers import CallbackShow, CallbackPrintIteration
from odl.discr import (Gradient, uniform_discr, ResizingOperator)
from odl.trafos import FourierTransform
from odl.phantom import (white_noise)
from odl.deform.linearized import _linear_deform
from odl.deform.mass_preserving import geometric_deform, mass_presv_deform
from odl.deform.LDDMM_gradient_descent_scheme import (padded_ft_op, 
                                                      fitting_kernel)
standard_library.install_aliases()


__all__ = ('LDDMM_gradient_descent_solver_spatiotemporal')


def LDDMM_gradient_descent_solver_spatiotemporal(
    forward_op, noise_proj_data, template, vector_fields, gate_pts, discr_deg, 
    niter, in_niter1, in_inter2, stepsize1, stepsize2, mu_1, mu_2, lamb, 
    kernel, impl1='geom', impl2='least_square', callback=None):
    """
    Solver for spatiotemporal image reconstruction using LDDMM.

    Notes
    -----
    The model is:
                
        .. math:: \min_{v} \lambda * \int_0^1 \|v(t)\|_V^2 dt + \|T(\phi_1.I) - g\|_2^2,

    where :math:`\phi_1.I := |D\phi_1^{-1}| I(\phi_1^{-1})` is for
    mass-preserving deformation, instead, :math:`\phi_1.I := I(\phi_1^{-1})`
    is for geometric deformation. :math:`\phi_1^{-1}` is the inverse of
    the solution at :math:`t=1` of flow of doffeomorphisms.
    :math:`|D\phi_1^{-1}|` is the Jacobian determinant of :math:`\phi_1^{-1}`.
    :math:`T` is the forward operator. If :math:`T` is an identity operator,
    the above model reduces to image matching. If :math:`T` is a non-identity
    forward operator, the above model is for shape-based image reconstrction.  
    :math:`g` is the detected data, `data_elem`. :math:`I` is the `template`.
    :math:`v(t)` is the velocity vector. :math:`V` is a reproducing kernel
    Hilbert space for velocity vector. :math:`lamb` is the regularization
    parameter. 
    

    Parameters
    ----------
    forward_op : `Operator`
        The forward operator of imaging.
    data_elem : `DiscreteLpElement`
        The given data.
    template : `DiscreteLpElement`
        Fixed template deformed by the deformation.
    time_pts : `int`
        The number of time intervals
    iter : `int`
        The given maximum iteration number.
    eps : `float`
        The given step size.
    lamb : `float`
        The given regularization parameter. It's a weighted value on the
        regularization-term side.
    kernel : `function`
        Kernel function in reproducing kernel Hilbert space.
    impl1 : {'geom', 'mp'}, optional
        The given implementation method for group action.
        The impl1 chooses 'mp' or 'geom', where 'mp' means using
        mass-preserving method, and 'geom' means using
        non-mass-preserving geometric method. Its defalt choice is 'geom'.
    impl2 : {'least square'}, optional
        The given implementation method for data matching term.
        Here the implementation only supports the case of least square.    
    callback : `class`, optional
        Show the intermediate results of iteration.

    Returns
    -------
    image_N0 : `ProductSpaceElement`
        The series of images produced by template and velocity field.
    mp_deformed_image_N0 : `ProductSpaceElement`
        The series of mass-preserving images produced by template
        and velocity field.
    E : `numpy.array`
        Storage of the energy values for iterations.
    """

    # Max index of discretized points
    N = gate_pts

    # Discretized degree
    M = discr_deg

    # Compute the max index of discretized points
    MN = M*N
    MN1 = MN + 1
    
    # Get the inverse of the number of discretized points
    inv_MN = 1.0 / MN
    
    N1 = N + 1
    N2 = 2. / N
    ss1 = stepsize1 * N2
    ss2 = stepsize2 * N2
    ss3 = stepsize1 * mu_1
    
    # Create the gradient operator for the squared L2 functional
    if impl2=='least_square':
        gradS = [forward_op[0].adjoint * (forward_op[0] - noise_proj_data[0])] * N1
        for i in range(N):
            j = i+1
            gradS[j] = forward_op[j].adjoint * (forward_op[j] - noise_proj_data[j])
    else:
        raise NotImplementedError('now only support least square')

    # Create the space of images
    image_space = template.space

    # Get the dimension of the space of images
    dim = image_space.ndim
    
    # Fourier transform setting for data matching term
    # The padded_size is the size of the padded domain 
    padded_size = 2 * image_space.shape[0]
    # The pad_ft_op is the operator of Fourier transform
    # composing with padded operator
    pad_ft_op = padded_ft_op(image_space, padded_size)
    # The vectorial_ft_op is a vectorial Fourier transform operator,
    # which constructs the diagnal element of a matrix.
    vectorial_ft_op = DiagonalOperator(*([pad_ft_op] * dim))
    
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(image_space, kernel)
    ft_kernel_fitting = vectorial_ft_op(discretized_kernel)

    # Create the space for series deformations and series Jacobian determinant
    series_image_space = ProductSpace(image_space, MN1)
    
    series_backprojection_space = ProductSpace(image_space, N1)
    series_bp_all = [image_space.element()] * N1
    
    for i in range(N):
        j = i + 1
        series_bp_all[j] = [image_space.element()] * (j*M+1)
    
    # Initialize vector fileds at different time points
    vector_fields = vector_fields

    # Initialize two series deformations and series Jacobian determinant
    image_MN0 = series_image_space.element()

    if impl1=='geom':
        eta_tt = series_backprojection_space.element()
    else:
        raise NotImplementedError('unknown group action')

    for j in range(MN1):
        image_MN0[j] = image_space.element(template)
    
    eta_tt[0] = gradS[0](image_MN0[0])
    series_bp_all[0] = eta_tt[0]

    for i in range(1, N1):
        iM = i*M
        eta_tt[i] = gradS[i](image_MN0[iM])
        for j in range(iM+1):
            series_bp_all[i][j] = eta_tt[i] 

    # Create the gradient operator
    grad_op = Gradient(domain=image_space, method='forward',
                       pad_mode='symmetric')

    # Create the divergence operator, which can be obtained from
    # the adjoint of gradient operator 
    # div_op = Divergence(domain=pspace, method='forward', pad_mode='symmetric')
    grad_op_adjoint = grad_op.adjoint
    div_op = - grad_op.adjoint

    # Begin iteration for non-mass-preserving case
    if impl1=='geom':
        print(impl1)
        # Outer iteration
        for k in range(niter):
            print('iter = {!r}'.format(k))

#%%%Setting for getting a proper initial template

            # Inner iteration for updating template
            if k == 0:
                niter1 = 50
#                niter1 = in_niter1
            else:
                niter1 = in_niter1
            
#%%%Solving TV-L2 by Gradient Descent
            # Store energy
            E = []
            E = np.hstack((E, np.zeros(niter1)))
            
            for k1 in range(niter1):
                image_MN0[0] = template

                # Update partial of template    
                grad_template = grad_op(template)
                grad_template_norm = np.sqrt(grad_template[0]**2 + grad_template[1]**2 + 1.0e-12)
                
                E[k1] += mu_1 * np.asarray(grad_template_norm).sum() * template.space.cell_volume
                for i in range(1, N1):
                    E[k1] += 1. / N * np.asarray((forward_op[i](image_MN0[i*M]) - noise_proj_data[i])**2).sum() \
                        * noise_proj_data[0].space.cell_volume

                template = template - \
                    ss3 * grad_op_adjoint(grad_template/grad_template_norm)
                
                for j in range(MN):
                    temp1 = j + 1
                    # Update image_MN0
                    image_MN0[temp1] = image_space.element(
                            _linear_deform(image_MN0[j], 
                                           -inv_MN * vector_fields[temp1])) 
                    if temp1 % M == 0:
                        temp2 = temp1 // M
#                        print(temp1)
#                        print(temp2)
                        
                        # Update eta_tt
                        eta_tt[temp2] = gradS[temp2](image_MN0[temp1])  
#                        eta_tt[temp2].show('eta_tt[{!r}]'.format(temp2))
                        series_bp_all[temp2][temp1] = eta_tt[temp2]
                        # the above two lines can be combined into one
                        # series_bp_all[temp2][temp1] = gradS[temp2](image_MN0[temp1]) 
                        
                        for l in range(temp1):
                            jacobian_det = image_space.element(
                                1.0 + inv_MN * div_op(vector_fields[temp1-l-1]))
                            # Update eta_tau_tnp
                            series_bp_all[temp2][temp1-l-1] = \
                                jacobian_det * image_space.element(
                                    _linear_deform(
                                        series_bp_all[temp2][temp1-l], 
                                        inv_MN * vector_fields[temp1-l-1]))
                        # Update partial of template
                        template = template - \
                            ss1 * series_bp_all[temp2][0]

            for k2 in range(in_inter2):
                image_MN0[0] = template
                series_bp_all[0] = gradS[0](image_MN0[0])

                for j in range(MN):
                    temp1 = j + 1
                    # Update image_MN0
                    image_MN0[temp1] = image_space.element(
                            _linear_deform(image_MN0[j], 
                                           -inv_MN * vector_fields[temp1])) 
                    if temp1 % M == 0:
                        temp2 = temp1 // M
                        # Update eta_tt
                        eta_tt[temp2] = gradS[temp2](image_MN0[temp1])
                        series_bp_all[temp2][temp1] = eta_tt[temp2]
                        
                        for l in range(temp1):
                            jacobian_det = image_space.element(
                                1.0 + inv_MN * div_op(vector_fields[temp1-l-1]))
                            # Update eta_tau_t
                            series_bp_all[temp2][temp1-l-1] = \
                                jacobian_det * image_space.element(
                                    _linear_deform(
                                        series_bp_all[temp2][temp1-l], 
                                        inv_MN * vector_fields[temp1-l-1]))
                
                for j in range(MN1):
                    tmp1 = grad_op(image_MN0[j])
                    tmp2 = int(np.ceil(j*1./M))
                    tmp0 = tmp2 + 1
#                    print(tmp2)
                    if tmp2 == 0:
                        tmp3 = image_space.zero()
                        tmp4 = image_space.tangent_bundle.zero()
                    else:
                        tmp3 = series_bp_all[tmp2][j]
                        tmp4 = vector_fields[j]
                    
                    for i in range(tmp0, N1):
                        tmp3 = tmp3 + series_bp_all[i][j]
                        tmp4 = tmp4 + vector_fields[j]

                    for i in range(dim):
                        tmp1[i] *= tmp3
                
                    tmp5 = (2 * np.pi) ** (dim / 2.0) * vectorial_ft_op.inverse(
                        vectorial_ft_op(tmp1) * ft_kernel_fitting)
                    # Update vector_fields
                    vector_fields[j] = vector_fields[j] + ss2 * (tmp5 - 
                        mu_2 * tmp4)

        return template, vector_fields, image_MN0
    else:
        raise NotImplementedError('unknown group action')
