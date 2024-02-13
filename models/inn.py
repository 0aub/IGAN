import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from typing import Callable, Iterable, List, Tuple, Union
import numpy as np
import warnings

# reference: https://github.com/vislearn/FrEIA

def output_dims_compatible(invertible_module):
    """
    Hack to get output dimensions from any module as
    SequenceINN and GraphINN do not work with input/output shape API.
    """
    no_output_dims = (
            hasattr(invertible_module, "force_tuple_output")
            and not invertible_module.force_tuple_output
    )
    if not no_output_dims:
        return invertible_module.output_dims(invertible_module.dims_in)
    else:
        try:
            return invertible_module.output_dims(None)
        except TypeError:
            raise NotImplementedError(f"Can't determine output dimensions for {invertible_module.__class__}.")



def list_of_int_tuples(list_of_tuples: List[Tuple[int]]) -> List[Tuple[int]]:
    BASIC_ERROR = (
        f"Invalid dimension specification: You passed {list_of_tuples}, but a "
        f"list of int tuples was expected. Problem:"
    )

    # Check that outermost list is iterable
    try:
        iter(list_of_tuples)
    except TypeError:
        raise TypeError(
             f"{BASIC_ERROR} {list_of_tuples!r} cannot be iterated."
        )

    # Check that each entry in list is iterable and contains only int-likes
    for int_tuple in list_of_tuples:
        try:
            iter(int_tuple)
        except TypeError:
            try:
                int(int_tuple)
                addendum = (
                    " Even if you have only one input, "
                    "you need to wrap it in a list."
                )
            except TypeError:
                addendum = ""
            raise TypeError(
                f"{BASIC_ERROR} {int_tuple!r} cannot be iterated.{addendum}"
            )
        for supposed_int in int_tuple:
            try:
                int(supposed_int)
            except TypeError:
                raise TypeError(
                    f"{BASIC_ERROR} {supposed_int!r} cannot be cast to an int."
                )

    return [tuple(map(int, int_tuple)) for int_tuple in list_of_tuples]


class InvertibleModule(nn.Module):
    """
    Base class for all invertible modules in FrEIA.

    Given ``module``, an instance of some InvertibleModule.
    This ``module`` shall be invertible in its input dimensions,
    so that the input can be recovered by applying the module
    in backwards mode (``rev=True``), not to be confused with
    ``pytorch.backward()`` which computes the gradient of an operation::

        x = torch.randn(BATCH_SIZE, DIM_COUNT)
        c = torch.randn(BATCH_SIZE, CONDITION_DIM)

        # Forward mode
        z, jac = module([x], [c], jac=True)

        # Backward mode
        x_rev, jac_rev = module(z, [c], rev=True)

    The ``module`` returns :math:`\\log \\det J = \\log \\left| \\det \\frac{\\partial f}{\\partial x} \\right|`
    of the operation in forward mode, and
    :math:`-\\log | \\det J | = \\log \\left| \\det \\frac{\\partial f^{-1}}{\\partial z} \\right| = -\\log \\left| \\det \\frac{\\partial f}{\\partial x} \\right|`
    in backward mode (``rev=True``).

    Then, ``torch.allclose(x, x_rev) == True`` and ``torch.allclose(jac, -jac_rev) == True``.
    """

    def __init__(self, dims_in: List[Tuple[int]],
                 dims_c: List[Tuple[int]] = None):
        """
        Args:
            dims_in: list of tuples specifying the shape of the inputs to this
                     operator: ``dims_in = [shape_x_0, shape_x_1, ...]``
            dims_c:  list of tuples specifying the shape of the conditions to
                     this operator.
        """
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = list_of_int_tuples(dims_in)
        self.dims_c = list_of_int_tuples(dims_c)

    def forward(self, x_or_z: Iterable[torch.Tensor], c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True) \
            -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        """
        Perform a forward (default, ``rev=False``) or backward pass (``rev=True``)
        through this module/operator.

        **Note to implementers:**

        - Subclasses MUST return a Jacobian when ``jac=True``, but CAN return a
          valid Jacobian when ``jac=False`` (not punished). The latter is only recommended
          if the computation of the Jacobian is trivial.
        - Subclasses MUST follow the convention that the returned Jacobian be
          consistent with the evaluation direction. Let's make this more precise:
          Let :math:`f` be the function that the subclass represents. Then:

          .. math::

              J &= \\log \\det \\frac{\\partial f}{\\partial x} \\\\
              -J &= \\log \\det \\frac{\\partial f^{-1}}{\\partial z}.

          Any subclass MUST return :math:`J` for forward evaluation (``rev=False``),
          and :math:`-J` for backward evaluation (``rev=True``).

        Args:
            x_or_z: input data (array-like of one or more tensors)
            c:      conditioning data (array-like of none or more tensors)
            rev:    perform backward pass
            jac:    return Jacobian associated to the direction
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide forward(...) method")

    def log_jacobian(self, *args, **kwargs):
        '''This method is deprecated, and does nothing except raise a warning.'''
        raise DeprecationWarning("module.log_jacobian(...) is deprecated. "
                                 "module.forward(..., jac=True) returns a "
                                 "tuple (out, jacobian) now.")

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        '''
        Used for shape inference during construction of the graph. MUST be
        implemented for each subclass of ``InvertibleModule``.

        Args:
          input_dims: A list with one entry for each input to the module.
            Even if the module only has one input, must be a list with one
            entry. Each entry is a tuple giving the shape of that input,
            excluding the batch dimension. For example for a module with one
            input, which receives a 32x32 pixel RGB image, ``input_dims`` would
            be ``[(3, 32, 32)]``

        Returns:
            A list structured in the same way as ``input_dims``. Each entry
            represents one output of the module, and the entry is a tuple giving
            the shape of that output. For example if the module splits the image
            into a right and a left half, the return value should be
            ``[(3, 16, 32), (3, 16, 32)]``. It is up to the implementor of the
            subclass to ensure that the total number of elements in all inputs
            and all outputs is consistent.
        '''
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide output_dims(...)")


class Inverse(InvertibleModule):
    """
    An invertible module that inverses a given module.
    """
    def __init__(self, module: InvertibleModule):
        # Hack as SequenceINN and GraphINN do not work with input/output shape API
        input_dims = output_dims_compatible(module)
        super().__init__(input_dims, module.dims_c)
        self.module = module

    @property
    def force_tuple_output(self):
        try:
            return self.module.force_tuple_output
        except AttributeError:
            return True

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return self.module.dims_in

    def forward(self, *args,
                rev: bool = False, **kwargs) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        return self.module(*args, rev=not rev, **kwargs)


class SequenceINN(InvertibleModule):
    """
    Simpler than FrEIA.framework.GraphINN:
    Only supports a sequential series of modules (no splitting, merging,
    branching off).
    Has an append() method, to add new blocks in a more simple way than the
    computation-graph based approach of GraphINN. For example:

    .. code-block:: python

       inn = SequenceINN(channels, dims_H, dims_W)

       for i in range(n_blocks):
           inn.append(FrEIA.modules.AllInOneBlock, clamp=2.0, permute_soft=True)
       inn.append(FrEIA.modules.HaarDownsampling)
       # and so on
    """

    def __init__(self, *dims: int, force_tuple_output=False):
        super().__init__([dims])

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.ModuleList()

        self.force_tuple_output = force_tuple_output

    def append(self, module_class, cond=None, cond_shape=None, **kwargs):
        """
        Append a reversible block from FrEIA.modules to the network.

        Args:
          module_class: Class from FrEIA.modules.
          cond (int): index of which condition to use (conditions will be passed as list to forward()).
            Conditioning nodes are not needed for SequenceINN.
          cond_shape (tuple[int]): the shape of the condition tensor.
          **kwargs: Further keyword arguments that are passed to the constructor of module_class (see example).
        """

        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)

        if isinstance(module_class, InvertibleModule):
            module = module_class
            if module.dims_in != dims_in:
                raise ValueError(
                    f"You passed an instance of {module.__class__.__name__} to "
                    f"SequenceINN which expects a {module.dims_in} input, "
                    f"but the output of the previous layer is of shape "
                    f"{dims_in}."
                )
            if len(kwargs) > 0:
                raise ValueError(
                    "You try to append an instanciated "
                    "InvertibleModule to SequenceINN, but also provided "
                    "constructor kwargs."
                )
        else:
            if cond is not None:
                kwargs['dims_c'] = [cond_shape]
            module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        output_dims = module.output_dims(dims_in)
        if len(output_dims) != 1:
            raise ValueError(
                f"Module of type {module.__class__} has more than one output: "
                f"{output_dims}"
            )
        self.shapes.append(output_dims[0])

    def __setitem__(self, key, value: InvertibleModule):
        """
        Replaces the module at position key with value.
        """
        if isinstance(key, slice):
            raise NotImplementedError("Setting sequence_inn[...] with slices as index is not supported.")
        existing_module = self.module_list[key]
        assert isinstance(existing_module, InvertibleModule)

        # Input dims
        if existing_module.dims_in != value.dims_in:
            raise ValueError(
                f"Module at position {key} must have input shape {existing_module.dims_in}, "
                f"but the replacement has input shape {value.dims_in}."
            )

        # Output dims
        existing_dims_out = existing_module.output_dims(existing_module.dims_in)
        target_dims_out = value.output_dims(value.dims_in)
        if existing_dims_out != target_dims_out:
            raise ValueError(
                f"Module at position {key} must have input shape {existing_dims_out}, "
                f"but the replacement has input shape {target_dims_out}."
            )

        # Condition
        if existing_module.dims_c != value.dims_c:
            raise ValueError(
                f"Module at position {key} must have condition shape {existing_dims_out}, "
                f"but the replacement has condition shape {target_dims_out}."
            )

        # Actually replace
        self.module_list[key] = value

    def __getitem__(self, item) -> Union[InvertibleModule, "SequenceINN"]:
        if isinstance(item, slice):
            # Zero-length
            in_dims = self.shapes[item]
            start, stop, stride = item.indices(len(self))
            sub_inn = SequenceINN(*self.shapes[start], force_tuple_output=self.force_tuple_output)
            if len(in_dims) == 0:
                return sub_inn
            cond_map = {None: None}
            cond_counter = 0
            for idx in range(start, stop, stride):
                module = self.module_list[idx]
                module_condition = self.conditions[idx]
                if stride < 0:
                    module = Inverse(module)
                if module_condition not in cond_map:
                    cond_map[module_condition] = cond_counter
                    cond_counter += 1
                sub_inn.append(module, cond_map[module_condition])
            return sub_inn

        return self.module_list.__getitem__(item)

    def __len__(self):
        return self.module_list.__len__()

    def __iter__(self):
        return self.module_list.__iter__()

    def output_dims(self, input_dims: List[Tuple[int]] = None) \
            -> List[Tuple[int]]:
        """
        Extends the definition in InvertibleModule to also return the output
        dimension when
        """
        if input_dims is not None:
            if self.force_tuple_output:
                if input_dims != self.shapes[0]:
                    raise ValueError(f"Passed input shapes {input_dims!r} do "
                                     f"not match with those passed in the "
                                     f"construction of the SequenceINN "
                                     f"{self.shapes[0]}")
            else:
                raise ValueError("You can only call output_dims on a "
                                 "SequenceINN when setting "
                                 "force_tuple_output=True.")
        return [self.shapes[-1]]

    def forward(self, x_or_z: torch.Tensor, c: Iterable[torch.Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the sequential INN in forward or inverse (rev=True) direction.

        Args:
            x_or_z: input tensor (in contrast to GraphINN, a list of
                    tensors is not supported, as SequenceINN only has
                    one input).
            c: list of conditions.
            rev: whether to compute the network forward or reversed.
            jac: whether to compute the log jacobian

        Returns:
            z_or_x (Tensor): network output.
            jac (Tensor): log-jacobian-determinant.
        """

        iterator = range(len(self.module_list))
        log_det_jac = torch.zeros(x_or_z.shape[0], device=x_or_z.device)

        if rev:
            iterator = reversed(iterator)

        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)
        for i in iterator:
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            else:
                x_or_z, j = self.module_list[i](x_or_z,
                                                c=[c[self.conditions[i]]],
                                                jac=jac, rev=rev)
            log_det_jac = j + log_det_jac

        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac


class AllInOneBlock(InvertibleModule):
    '''Module combining the most common operations in a normalizing flow or similar model.

    It combines affine coupling, permutation, and global affine transformation
    ('ActNorm'). It can also be used as GIN coupling block, perform learned
    householder permutations, and use an inverted pre-permutation. The affine
    transformation includes a soft clamping mechanism, first used in Real-NVP.
    The block as a whole performs the following computation:

    .. math::

        y = V\\,R \\; \\Psi(s_\\mathrm{global}) \\odot \\mathrm{Coupling}\\Big(R^{-1} V^{-1} x\\Big)+ t_\\mathrm{global}

    - The inverse pre-permutation of x (i.e. :math:`R^{-1} V^{-1}`) is optional (see
      ``reverse_permutation`` below).
    - The learned householder reflection matrix
      :math:`V` is also optional all together (see ``learned_householder_permutation``
      below).
    - For the coupling, the input is split into :math:`x_1, x_2` along
      the channel dimension. Then the output of the coupling operation is the
      two halves :math:`u = \\mathrm{concat}(u_1, u_2)`.

      .. math::

          u_1 &= x_1 \\odot \\exp \\Big( \\alpha \\; \\mathrm{tanh}\\big( s(x_2) \\big)\\Big) + t(x_2) \\\\
          u_2 &= x_2

      Because :math:`\\mathrm{tanh}(s) \\in [-1, 1]`, this clamping mechanism prevents
      exploding values in the exponential. The hyperparameter :math:`\\alpha` can be adjusted.

    '''

    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor: Callable = None,
                 affine_clamping: float = 2.,
                 gin_block: bool = False,
                 global_affine_init: float = 1.,
                 global_affine_type: str = 'SOFTPLUS',
                 permute_soft: bool = False,
                 learned_householder_permutation: int = 0,
                 reverse_permutation: bool = False):
        '''
        Args:
          subnet_constructor:
            class or callable ``f``, called as ``f(channels_in, channels_out)`` and
            should return a torch.nn.Module. Predicts coupling coefficients :math:`s, t`.
          affine_clamping:
            clamp the output of the multiplicative coefficients before
            exponentiation to +/- ``affine_clamping`` (see :math:`\\alpha` above).
          gin_block:
            Turn the block into a GIN block from Sorrenson et al, 2019.
            Makes it so that the coupling operations as a whole is volume preserving.
          global_affine_init:
            Initial value for the global affine scaling :math:`s_\mathrm{global}`.
          global_affine_init:
            ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation to be used
            on the beta for the global affine scaling (:math:`\\Psi` above).
          permute_soft:
            bool, whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
            or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
            when working with >512 dimensions.
          learned_householder_permutation:
            Int, if >0, turn on the matrix :math:`V` above, that represents
            multiple learned householder reflections. Slow if large number.
            Dubious whether it actually helps network performance.
          reverse_permutation:
            Reverse the permutation before the block, as introduced by Putzky
            et al, 2019. Turns on the :math:`R^{-1} V^{-1}` pre-multiplication above.
        '''

        super().__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            if permute_soft or learned_householder_permutation:
                self.permute_function = {0: F.linear,
                                        1: F.conv1d,
                                        2: F.conv2d,
                                        3: F.conv3d}[self.input_rank]
            else:
                self.permute_function = lambda x, p: x[:, p]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        self.in_channels         = channels
        self.clamp               = affine_clamping
        self.GIN                 = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder         = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(("Soft permutation will take a very long time to initialize "
                           f"with {channels} feature channels. Consider using hard permutation instead."))

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == 'SIGMOID':
            global_scale = 2. - np.log(10. / global_affine_init - 1.)
            self.global_scale_activation = self._sigmoid_global_scale_activation
        elif global_affine_type == 'SOFTPLUS':
            global_scale = 2. * np.log(np.exp(0.5 * 10. * global_affine_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = self._softplus_global_scale_activation
        elif global_affine_type == 'EXP':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = self._exp_global_scale_activation
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.global_scale = nn.Parameter(torch.ones(1, self.in_channels, *([1] * self.input_rank)) * float(global_scale))
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels, *([1] * self.input_rank)))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w_index = torch.randperm(channels, requires_grad=False)

        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.from_numpy(w).float(), requires_grad=False)
        elif permute_soft:
            self.w_perm = nn.Parameter(torch.from_numpy(w).float().view(channels, channels, *([1] * self.input_rank)).contiguous(),
                                       requires_grad=False)
            self.w_perm_inv = nn.Parameter(torch.from_numpy(w.T).float().view(channels, channels, *([1] * self.input_rank)).contiguous(),
                                           requires_grad=False)
        else:
            self.w_perm = nn.Parameter(w_index, requires_grad=False)
            self.w_perm_inv = nn.Parameter(torch.argsort(w_index), requires_grad=False)

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor "
                             "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

    def _sigmoid_global_scale_activation(self, a):
        return 10 * torch.sigmoid(a - 2.)

    def _softplus_global_scale_activation(self, a):
        return 0.1 * self.softplus(a)

    def _exp_global_scale_activation(self, a):
        return torch.exp(a)

    def _construct_householder_permutation(self):
        '''Computes a permutation matrix from the reflection vectors that are
        learned internally as nn.Parameters.'''
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for i in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''
        if self.GIN:
            scale = 1.
            perm_log_jac = 0.
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale,
                    perm_log_jac)
        else:
            return (self.permute_function(x * scale + self.global_offset, self.w_perm),
                    perm_log_jac)

    def _pre_permute(self, x, rev=False):
        '''Permutes before the coupling block, only used if
        reverse_permutation is set'''
        if rev:
            return self.permute_function(x, self.w_perm)
        else:
            return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x, a, rev=False):
        '''Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet.'''

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a = a * 0.1
        ch = x.shape[1]

        sub_jac = self.clamp * torch.tanh(a[:, :ch]/self.clamp)
        if self.GIN:
            sub_jac = sub_jac - torch.mean(sub_jac, dim=self.sum_dims, keepdim=True) 

        if not rev:
            return (x * torch.exp(sub_jac) + a[:, ch:],
                    torch.sum(sub_jac, dim=self.sum_dims))
        else:
            return ((x - a[:, ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=self.sum_dims))

    def forward(self, x, c=[], rev=False, jac=True):
        '''See base class docstring'''
        if tuple(x[0].shape[1:]) != self.dims_in[0]:
            raise RuntimeError(f"Expected input of shape {self.dims_in[0]}, "
                             f"got {tuple(x[0].shape[1:])}.")
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det = log_jac_det + (-1)**rev * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class PermuteRandom(InvertibleModule):
    '''Constructs a random permutation, that stays fixed during training.
    Permutes along the first (channel-) dimension for multi-dimenional tensors.'''

    def __init__(self, dims_in, dims_c=None, seed: Union[int, None] = None):
        '''Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        '''
        super().__init__(dims_in, dims_c)

        self.in_channels = dims_in[0][0]

        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = nn.Parameter(torch.LongTensor(self.perm), requires_grad=False)
        self.perm_inv = nn.Parameter(torch.LongTensor(self.perm_inv), requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        if not rev:
            return [x[0][:, self.perm]], 0.
        else:
            return [x[0][:, self.perm_inv]], 0.

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims