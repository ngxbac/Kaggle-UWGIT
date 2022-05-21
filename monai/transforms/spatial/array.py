# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided

from monai.config import USE_COMPILED, DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, reorient_spatial_axes, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter, grid_pull
from monai.networks.utils import meshgrid_ij, normalize_transform
from monai.transforms.croppad.array import CenterSpatialCrop, Pad
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.transform import Randomizable, RandomizableTransform, ThreadUnsafe, Transform
from monai.transforms.utils import (
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
    map_spatial_axes,
)
from monai.transforms.utils_pytorch_numpy_unification import allclose, moveaxis
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
    pytorch_after,
)
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.enums import TransformBackends
from monai.utils.misc import ImageMetaKey as Key
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

nib, has_nib = optional_import("nibabel")

__all__ = [
    "SpatialResample",
    "ResampleToMatch",
    "Spacing",
    "Orientation",
    "Flip",
    "GridDistortion",
    "GridSplit",
    "Resize",
    "Rotate",
    "Zoom",
    "Rotate90",
    "RandRotate90",
    "RandRotate",
    "RandFlip",
    "RandGridDistortion",
    "RandAxisFlip",
    "RandZoom",
    "AffineGrid",
    "RandAffineGrid",
    "RandDeformGrid",
    "Resample",
    "Affine",
    "RandAffine",
    "Rand2DElastic",
    "Rand3DElastic",
]

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]


class SpatialResample(Transform):
    """
    Resample input image from the orientation/spacing defined by ``src_affine`` affine matrix into
    the ones specified by ``dst_affine`` affine matrix.

    Internally this transform computes the affine transform matrix from ``src_affine`` to ``dst_affine``,
    by ``xform = linalg.solve(src_affine, dst_affine)``, and call ``monai.transforms.Affine`` with ``xform``.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
    ):
        """
        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        src_affine: Optional[NdarrayOrTensor] = None,
        dst_affine: Optional[NdarrayOrTensor] = None,
        spatial_size: Optional[Union[Sequence[int], np.ndarray, int]] = None,
        mode: Union[GridSampleMode, str, None] = None,
        padding_mode: Union[GridSamplePadMode, str, None] = None,
        align_corners: Optional[bool] = False,
        dtype: DtypeLike = None,
    ) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
        """
        Args:
            img: input image to be resampled. It currently supports channel-first arrays with
                at most three spatial dimensions.
            src_affine: source affine matrix. Defaults to ``None``, which means the identity matrix.
                the shape should be `(r+1, r+1)` where `r` is the spatial rank of ``img``.
            dst_affine: destination affine matrix. Defaults to ``None``, which means the same as `src_affine`.
                the shape should be `(r+1, r+1)` where `r` is the spatial rank of ``img``.
                when `dst_affine` and `spatial_size` are None, the input will be returned without resampling,
                but the data type will be `float32`.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined,
                the transform will compute a spatial size automatically containing the previous field of view.
                if `spatial_size` is ``-1`` are the transform will use the corresponding input img size.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype`` or
                ``np.float64`` (for best precision). If ``None``, use the data type of input data.
                To be compatible with other modules, the output data type is always `float32`.

        The spatial rank is determined by the smallest among ``img.ndim -1``, ``len(src_affine) - 1``, and ``3``.

        When both ``monai.config.USE_COMPILED`` and ``align_corners`` are set to ``True``,
        MONAI's resampling implementation will be used.
        Set `dst_affine` and `spatial_size` to `None` to turn off the resampling step.
        """
        if src_affine is None:
            src_affine = np.eye(4, dtype=np.float64)
        spatial_rank = min(len(img.shape) - 1, src_affine.shape[0] - 1, 3)
        if (not isinstance(spatial_size, int) or spatial_size != -1) and spatial_size is not None:
            spatial_rank = min(len(ensure_tuple(spatial_size)), 3)  # infer spatial rank based on spatial_size
        src_affine = to_affine_nd(spatial_rank, src_affine)
        dst_affine = to_affine_nd(spatial_rank, dst_affine) if dst_affine is not None else src_affine
        dst_affine, *_ = convert_to_dst_type(dst_affine, dst_affine, dtype=torch.float32)

        in_spatial_size = np.asarray(img.shape[1 : spatial_rank + 1])
        if isinstance(spatial_size, int) and (spatial_size == -1):  # using the input spatial size
            spatial_size = in_spatial_size
        elif spatial_size is None and spatial_rank > 1:  # auto spatial size
            spatial_size, _ = compute_shape_offset(in_spatial_size, src_affine, dst_affine)  # type: ignore
        spatial_size = np.asarray(fall_back_tuple(ensure_tuple(spatial_size)[:spatial_rank], in_spatial_size))

        if (
            allclose(src_affine, dst_affine, atol=AFFINE_TOL)
            and allclose(spatial_size, in_spatial_size)
            or spatial_rank == 1
        ):
            # no significant change, return original image
            output_data, *_ = convert_to_dst_type(img, img, dtype=torch.float32)
            return output_data, dst_affine

        if has_nib and isinstance(img, np.ndarray):
            spatial_ornt, dst_r = reorient_spatial_axes(img.shape[1 : spatial_rank + 1], src_affine, dst_affine)
            if allclose(dst_r, dst_affine, atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size):
                # simple reorientation achieves the desired affine
                spatial_ornt[:, 0] += 1
                spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
                img_ = nib.orientations.apply_orientation(img, spatial_ornt)
                output_data, *_ = convert_to_dst_type(img_, img, dtype=torch.float32)
                return output_data, dst_affine

        try:
            src_affine, *_ = convert_to_dst_type(src_affine, dst_affine)
            if isinstance(src_affine, np.ndarray):
                xform = np.linalg.solve(src_affine, dst_affine)
            else:
                xform = (
                    torch.linalg.solve(src_affine, dst_affine)
                    if pytorch_after(1, 8, 0)
                    else torch.solve(dst_affine, src_affine).solution  # type: ignore
                )
        except (np.linalg.LinAlgError, RuntimeError) as e:
            raise ValueError(f"src affine is not invertible: {src_affine}") from e
        xform = to_affine_nd(spatial_rank, xform)
        # no resampling if it's identity transform
        if allclose(xform, np.diag(np.ones(len(xform))), atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size):
            output_data, *_ = convert_to_dst_type(img, img, dtype=torch.float32)
            return output_data, dst_affine

        _dtype = dtype or self.dtype or img.dtype
        in_spatial_size = in_spatial_size.tolist()
        chns, additional_dims = img.shape[0], img.shape[spatial_rank + 1 :]  # beyond three spatial dims
        # resample
        img_ = convert_data_type(img, torch.Tensor, dtype=_dtype)[0]
        xform = convert_to_dst_type(xform, img_)[0]
        align_corners = self.align_corners if align_corners is None else align_corners
        mode = mode or self.mode
        padding_mode = padding_mode or self.padding_mode
        if additional_dims:
            xform_shape = [-1] + in_spatial_size
            img_ = img_.reshape(xform_shape)
        if align_corners:
            _t_r = torch.diag(torch.ones(len(xform), dtype=xform.dtype, device=xform.device))  # type: ignore
            for idx, d_dst in enumerate(spatial_size[:spatial_rank]):
                _t_r[idx, -1] = (max(d_dst, 2) - 1.0) / 2.0
            xform = xform @ _t_r
            if not USE_COMPILED:
                _t_l = normalize_transform(
                    in_spatial_size, xform.device, xform.dtype, align_corners=True  # type: ignore
                )
                xform = _t_l @ xform  # type: ignore
            affine_xform = Affine(
                affine=xform, spatial_size=spatial_size, normalized=True, image_only=True, dtype=_dtype
            )
            output_data = affine_xform(img_, mode=mode, padding_mode=padding_mode)
        else:
            affine_xform = AffineTransform(
                normalized=False,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                reverse_indexing=True,
            )
            output_data = affine_xform(img_.unsqueeze(0), theta=xform, spatial_size=spatial_size).squeeze(0)
        if additional_dims:
            full_shape = (chns, *spatial_size, *additional_dims)
            output_data = output_data.reshape(full_shape)
        # output dtype float
        output_data, *_ = convert_to_dst_type(output_data, img, dtype=torch.float32)
        return output_data, dst_affine


class ResampleToMatch(SpatialResample):
    """Resample an image to match given metadata. The affine matrix will be aligned,
    and the size of the output image will match."""

    def __call__(  # type: ignore
        self,
        img: NdarrayOrTensor,
        src_meta: Optional[Dict] = None,
        dst_meta: Optional[Dict] = None,
        mode: Union[GridSampleMode, str, None] = None,
        padding_mode: Union[GridSamplePadMode, str, None] = None,
        align_corners: Optional[bool] = False,
        dtype: DtypeLike = None,
    ):
        if src_meta is None:
            raise RuntimeError("`in_meta` is missing")
        if dst_meta is None:
            raise RuntimeError("`out_meta` is missing")
        mode = mode or self.mode
        padding_mode = padding_mode or self.padding_mode
        align_corners = self.align_corners if align_corners is None else align_corners
        dtype = dtype or self.dtype
        src_affine = src_meta.get("affine")
        dst_affine = dst_meta.get("affine")
        img, updated_affine = super().__call__(
            img=img,
            src_affine=src_affine,
            dst_affine=dst_affine,
            spatial_size=dst_meta.get("spatial_shape"),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=dtype,
        )
        dst_meta = deepcopy(dst_meta)
        dst_meta["affine"] = updated_affine
        dst_meta[Key.FILENAME_OR_OBJ] = src_meta.get(Key.FILENAME_OR_OBJ)
        return img, dst_meta


class Spacing(Transform):
    """
    Resample input image into the specified `pixdim`.
    """

    backend = SpatialResample.backend

    def __init__(
        self,
        pixdim: Union[Sequence[float], float, np.ndarray],
        diagonal: bool = False,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: DtypeLike = np.float64,
        image_only: bool = False,
    ) -> None:
        """
        Args:
            pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
                items of the pixdim sequence map to the spatial dimensions of input image, if length
                of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
                if shorter, will pad with `1.0`.
                if the components of the `pixdim` are non-positive values, the transform will use the
                corresponding components of the original pixdim, which is computed from the `affine`
                matrix of input image.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
            image_only: return just the image or the image, the old affine and new affine. Default is `False`.

        """
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.image_only = image_only

        self.sp_resample = SpatialResample(
            mode=look_up_option(mode, GridSampleMode),
            padding_mode=look_up_option(padding_mode, GridSamplePadMode),
            align_corners=align_corners,
            dtype=dtype,
        )

    def __call__(
        self,
        data_array: NdarrayOrTensor,
        affine: Optional[NdarrayOrTensor] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: DtypeLike = None,
        output_spatial_shape: Optional[Union[Sequence[int], np.ndarray, int]] = None,
    ) -> Union[NdarrayOrTensor, Tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]]:
        """
        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
            output_spatial_shape: specify the shape of the output data_array. This is typically useful for
                the inverse of `Spacingd` where sometimes we could not compute the exact shape due to the quantization
                error with the affine.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``pixdim`` is nonpositive.

        Returns:
            data_array (resampled into `self.pixdim`), original affine, current affine.

        """
        sr = int(data_array.ndim - 1)
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            # default to identity
            affine_np = affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_np, *_ = convert_data_type(affine, np.ndarray)
            affine_ = to_affine_nd(sr, affine_np)

        out_d = self.pixdim[:sr]
        if out_d.size < sr:
            out_d = np.append(out_d, [1.0] * (sr - out_d.size))

        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        output_shape, offset = compute_shape_offset(data_array.shape[1:], affine_, new_affine)
        new_affine[:sr, -1] = offset[:sr]
        output_data, new_affine = self.sp_resample(
            data_array,
            src_affine=affine,
            dst_affine=new_affine,
            spatial_size=list(output_shape) if output_spatial_shape is None else output_spatial_shape,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            dtype=dtype,
        )
        new_affine = to_affine_nd(affine_np, new_affine)
        new_affine, *_ = convert_to_dst_type(src=new_affine, dst=affine, dtype=torch.float32)

        if self.image_only:
            return output_data
        return output_data, affine, new_affine


class Orientation(Transform):
    """
    Change the input image's orientation into the specified based on `axcodes`.
    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        axcodes: Optional[str] = None,
        as_closest_canonical: bool = False,
        labels: Optional[Sequence[Tuple[str, str]]] = (("L", "R"), ("P", "A"), ("I", "S")),
        image_only: bool = False,
    ) -> None:
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.
            image_only: if True return only the image volume, otherwise return (image, affine, new_affine).

        Raises:
            ValueError: When ``axcodes=None`` and ``as_closest_canonical=True``. Incompatible values.

        See Also: `nibabel.orientations.ornt2axcodes`.

        """
        if axcodes is None and not as_closest_canonical:
            raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
        if axcodes is not None and as_closest_canonical:
            warnings.warn("using as_closest_canonical=True, axcodes ignored.")
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels
        self.image_only = image_only

    def __call__(
        self, data_array: NdarrayOrTensor, affine: Optional[NdarrayOrTensor] = None
    ) -> Union[NdarrayOrTensor, Tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor]]:
        """
        original orientation of `data_array` is defined by `affine`.

        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``axcodes`` spatiality differs from ``data_array``.

        Returns:
            data_array [reoriented in `self.axcodes`] if `self.image_only`, else
            (data_array [reoriented in `self.axcodes`], original axcodes, current axcodes).

        """
        spatial_shape = data_array.shape[1:]
        sr = len(spatial_shape)
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        affine_: np.ndarray
        if affine is None:
            # default to identity
            affine_np = affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_np, *_ = convert_data_type(affine, np.ndarray)
            affine_ = to_affine_nd(sr, affine_np)

        src = nib.io_orientation(affine_)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            if self.axcodes is None:
                raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
            if sr < len(self.axcodes):
                warnings.warn(
                    f"axcodes ('{self.axcodes}') length is smaller than the number of input spatial dimensions D={sr}.\n"
                    f"{self.__class__.__name__}: input spatial shape is {spatial_shape}, num. channels is {data_array.shape[0]},"
                    "please make sure the input is in the channel-first format."
                )
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
            if len(dst) < sr:
                raise ValueError(
                    f"axcodes must match data_array spatially, got axcodes={len(self.axcodes)}D data_array={sr}D"
                )
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        new_affine = affine_ @ nib.orientations.inv_ornt_aff(spatial_ornt, spatial_shape)
        _is_tensor = isinstance(data_array, torch.Tensor)
        spatial_ornt[:, 0] += 1  # skip channel dim
        spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
        axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
        if axes:
            data_array = (
                torch.flip(data_array, dims=axes) if _is_tensor else np.flip(data_array, axis=axes)  # type: ignore
            )
        full_transpose = np.arange(len(data_array.shape))
        full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
        if not np.all(full_transpose == np.arange(len(data_array.shape))):
            if _is_tensor:
                data_array = data_array.permute(full_transpose.tolist())  # type: ignore
            else:
                data_array = data_array.transpose(full_transpose)  # type: ignore
        out, *_ = convert_to_dst_type(src=data_array, dst=data_array)
        new_affine = to_affine_nd(affine_np, new_affine)
        new_affine, *_ = convert_to_dst_type(src=new_affine, dst=affine, dtype=torch.float32)

        if self.image_only:
            return out
        return out, affine, new_affine


class Flip(Transform):
    """
    Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html.

    Args:
        spatial_axis: spatial axes along which to flip over. Default is None.
            The default `axis=None` will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        self.spatial_axis = spatial_axis

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        if isinstance(img, np.ndarray):
            return np.ascontiguousarray(np.flip(img, map_spatial_axes(img.ndim, self.spatial_axis)))
        return torch.flip(img, map_spatial_axes(img.ndim, self.spatial_axis))


class Resize(Transform):
    """
    Resize the input image to given spatial size (with scaling, not cropping/padding).
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    Args:
        spatial_size: expected shape of spatial dimensions after resize operation.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        size_mode: should be "all" or "longest", if "all", will use `spatial_size` for all the spatial dims,
            if "longest", rescale the image so that only the longest side is equal to specified `spatial_size`,
            which must be an int number in this case, keeping the aspect ratio of the initial image, refer to:
            https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/
            #albumentations.augmentations.geometric.resize.LongestMaxSize.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        anti_aliasing: bool = False,
        anti_aliasing_sigma: Union[Sequence[float], float, None] = None,
    ) -> None:
        self.size_mode = look_up_option(size_mode, ["all", "longest"])
        self.spatial_size = spatial_size
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        align_corners: Optional[bool] = None,
        anti_aliasing: Optional[bool] = None,
        anti_aliasing_sigma: Union[Sequence[float], float, None] = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            anti_aliasing: bool, optional
                Whether to apply a Gaussian filter to smooth the image prior
                to downsampling. It is crucial to filter when downsampling
                the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
            anti_aliasing_sigma: {float, tuple of floats}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1. For the up-size case, s < 1, no
                anti-aliasing is performed prior to rescaling.

        Raises:
            ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

        """
        anti_aliasing = self.anti_aliasing if anti_aliasing is None else anti_aliasing
        anti_aliasing_sigma = self.anti_aliasing_sigma if anti_aliasing_sigma is None else anti_aliasing_sigma

        img_, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        if self.size_mode == "all":
            input_ndim = img_.ndim - 1  # spatial ndim
            output_ndim = len(ensure_tuple(self.spatial_size))
            if output_ndim > input_ndim:
                input_shape = ensure_tuple_size(img_.shape, output_ndim + 1, 1)
                img_ = img_.reshape(input_shape)
            elif output_ndim < input_ndim:
                raise ValueError(
                    "len(spatial_size) must be greater or equal to img spatial dimensions, "
                    f"got spatial_size={output_ndim} img={input_ndim}."
                )
            spatial_size_ = fall_back_tuple(self.spatial_size, img_.shape[1:])
        else:  # for the "longest" mode
            img_size = img_.shape[1:]
            if not isinstance(self.spatial_size, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
            scale = self.spatial_size / max(img_size)
            spatial_size_ = tuple(int(round(s * scale)) for s in img_size)

        if anti_aliasing and any(x < y for x, y in zip(spatial_size_, img_.shape[1:])):
            factors = torch.div(torch.Tensor(list(img_.shape[1:])), torch.Tensor(spatial_size_))
            if anti_aliasing_sigma is None:
                # if sigma is not given, use the default sigma in skimage.transform.resize
                anti_aliasing_sigma = torch.maximum(torch.zeros(factors.shape), (factors - 1) / 2).tolist()
            else:
                # if sigma is given, use the given value for downsampling axis
                anti_aliasing_sigma = list(ensure_tuple_rep(anti_aliasing_sigma, len(spatial_size_)))
                for axis in range(len(spatial_size_)):
                    anti_aliasing_sigma[axis] = anti_aliasing_sigma[axis] * int(factors[axis] > 1)
            anti_aliasing_filter = GaussianSmooth(sigma=anti_aliasing_sigma)
            img_ = anti_aliasing_filter(img_)

        resized = torch.nn.functional.interpolate(
            input=img_.unsqueeze(0),
            size=spatial_size_,
            mode=look_up_option(self.mode if mode is None else mode, InterpolateMode).value,
            align_corners=self.align_corners if align_corners is None else align_corners,
        )
        out, *_ = convert_to_dst_type(resized.squeeze(0), img)
        return out


class Rotate(Transform, ThreadUnsafe):
    """
    Rotates an input image by given angle using :py:class:`monai.networks.layers.AffineTransform`.

    Args:
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``np.float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        angle: Union[Sequence[float], float],
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: Union[DtypeLike, torch.dtype] = np.float32,
    ) -> None:
        self.angle = angle
        self.keep_size = keep_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype
        self._rotation_matrix: Optional[NdarrayOrTensor] = None

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: Union[DtypeLike, torch.dtype] = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.

        Raises:
            ValueError: When ``img`` spatially is not one of [2D, 3D].

        """
        _dtype = dtype or self.dtype or img.dtype

        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=_dtype)

        im_shape = np.asarray(img_t.shape[1:])  # spatial dimensions
        input_ndim = len(im_shape)
        if input_ndim not in (2, 3):
            raise ValueError(f"Unsupported img dimension: {input_ndim}, available options are [2, 3].")
        _angle = ensure_tuple_rep(self.angle, 1 if input_ndim == 2 else 3)
        transform = create_rotate(input_ndim, _angle)
        shift = create_translate(input_ndim, ((im_shape - 1) / 2).tolist())
        if self.keep_size:
            output_shape = im_shape
        else:
            corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape(
                (len(im_shape), -1)
            )
            corners = transform[:-1, :-1] @ corners  # type: ignore
            output_shape = np.asarray(corners.ptp(axis=1) + 0.5, dtype=int)
        shift_1 = create_translate(input_ndim, (-(output_shape - 1) / 2).tolist())
        transform = shift @ transform @ shift_1

        transform_t, *_ = convert_to_dst_type(transform, img_t)

        xform = AffineTransform(
            normalized=False,
            mode=look_up_option(mode or self.mode, GridSampleMode),
            padding_mode=look_up_option(padding_mode or self.padding_mode, GridSamplePadMode),
            align_corners=self.align_corners if align_corners is None else align_corners,
            reverse_indexing=True,
        )
        output: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=output_shape).float().squeeze(0)
        self._rotation_matrix = transform
        out: NdarrayOrTensor
        out, *_ = convert_to_dst_type(output, dst=img, dtype=output.dtype)
        return out

    def get_rotation_matrix(self) -> Optional[NdarrayOrTensor]:
        """
        Get the most recently applied rotation matrix
        This is not thread-safe.
        """
        return self._rotation_matrix


class Zoom(Transform):
    """
    Zooms an ND image using :py:class:`torch.nn.functional.interpolate`.
    For details, please see https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html.

    Different from :py:class:`monai.transforms.resize`, this transform takes scaling factors
    as input, and provides an option of preserving the input spatial size.

    Args:
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        keep_size: Should keep original size (padding/slicing if needed), default is True.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        zoom: Union[Sequence[float], float],
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        padding_mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
        **kwargs,
    ) -> None:
        self.zoom = zoom
        self.mode: InterpolateMode = InterpolateMode(mode)
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.keep_size = keep_size
        self.kwargs = kwargs

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

        """
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float32)

        _zoom = ensure_tuple_rep(self.zoom, img.ndim - 1)  # match the spatial image dim
        zoomed: NdarrayOrTensor = torch.nn.functional.interpolate(  # type: ignore
            recompute_scale_factor=True,
            input=img_t.unsqueeze(0),
            scale_factor=list(_zoom),
            mode=look_up_option(self.mode if mode is None else mode, InterpolateMode).value,
            align_corners=self.align_corners if align_corners is None else align_corners,
        )
        zoomed = zoomed.squeeze(0)

        if self.keep_size and not np.allclose(img_t.shape, zoomed.shape):

            pad_vec = [(0, 0)] * len(img_t.shape)
            slice_vec = [slice(None)] * len(img_t.shape)
            for idx, (od, zd) in enumerate(zip(img_t.shape, zoomed.shape)):
                diff = od - zd
                half = abs(diff) // 2
                if diff > 0:  # need padding
                    pad_vec[idx] = (half, diff - half)
                elif diff < 0:  # need slicing
                    slice_vec[idx] = slice(half, half + od)

            padder = Pad(pad_vec, padding_mode or self.padding_mode)
            zoomed = padder(zoomed)
            zoomed = zoomed[tuple(slice_vec)]

        out, *_ = convert_to_dst_type(zoomed, dst=img)
        return out


class Rotate90(Transform):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    See np.rot90 for additional details:
    https://numpy.org/doc/stable/reference/generated/numpy.rot90.html.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                If axis is negative it counts from the last to the first axis.
        """
        self.k = k
        spatial_axes_: Tuple[int, int] = ensure_tuple(spatial_axes)  # type: ignore
        if len(spatial_axes_) != 2:
            raise ValueError("spatial_axes must be 2 int numbers to indicate the axes to rotate 90 degrees.")
        self.spatial_axes = spatial_axes_

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        rot90: Callable = torch.rot90 if isinstance(img, torch.Tensor) else np.rot90  # type: ignore
        out: NdarrayOrTensor = rot90(img, self.k, map_spatial_axes(img.ndim, self.spatial_axes))
        out, *_ = convert_data_type(out, dtype=img.dtype)
        return out


class RandRotate90(RandomizableTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    backend = Rotate90.backend

    def __init__(self, prob: float = 0.1, max_k: int = 3, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        RandomizableTransform.__init__(self, prob)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._rand_k = self.R.randint(self.max_k) + 1

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        return Rotate90(self._rand_k, self.spatial_axes)(img)


class RandRotate(RandomizableTransform):
    """
    Randomly rotate the input arrays.

    Args:
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y).
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z).
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``np.float32``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
    """

    backend = Rotate.backend

    def __init__(
        self,
        range_x: Union[Tuple[float, float], float] = 0.0,
        range_y: Union[Tuple[float, float], float] = 0.0,
        range_z: Union[Tuple[float, float], float] = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        align_corners: bool = False,
        dtype: Union[DtypeLike, torch.dtype] = np.float32,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.keep_size = keep_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.align_corners = align_corners
        self.dtype = dtype

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        align_corners: Optional[bool] = None,
        dtype: Union[DtypeLike, torch.dtype] = None,
        randomize: bool = True,
        get_matrix: bool = False,
    ):
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
            randomize: whether to execute `randomize()` function first, default to True.
            get_matrix: whether to return the rotated image and rotate matrix together, default to False.
        """
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        rotator = Rotate(
            angle=self.x if img.ndim == 3 else (self.x, self.y, self.z),
            keep_size=self.keep_size,
            mode=look_up_option(mode or self.mode, GridSampleMode),
            padding_mode=look_up_option(padding_mode or self.padding_mode, GridSamplePadMode),
            align_corners=self.align_corners if align_corners is None else align_corners,
            dtype=dtype or self.dtype or img.dtype,
        )
        img = rotator(img)
        return (img, rotator.get_rotation_matrix()) if get_matrix else img


class RandFlip(RandomizableTransform):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        RandomizableTransform.__init__(self, prob)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize(None)

        if not self._do_transform:
            return img

        return self.flipper(img)


class RandAxisFlip(RandomizableTransform):
    """
    Randomly select a spatial axis and flip along it.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.

    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1) -> None:
        RandomizableTransform.__init__(self, prob)
        self._axis: Optional[int] = None

    def randomize(self, data: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._axis = self.R.randint(data.ndim - 1)

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if randomize:
            self.randomize(data=img)

        if not self._do_transform:
            return img

        return Flip(spatial_axis=self._axis)(img)


class RandZoom(RandomizableTransform):
    """
    Randomly zooms input arrays with given probability within given zoom range.

    Args:
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        keep_size: Should keep original size (pad if needed), default is True.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.

    """

    backend = Zoom.backend

    def __init__(
        self,
        prob: float = 0.1,
        min_zoom: Union[Sequence[float], float] = 0.9,
        max_zoom: Union[Sequence[float], float] = 1.1,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        padding_mode: Union[NumpyPadMode, PytorchPadMode, str] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
        **kwargs,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.min_zoom = ensure_tuple(min_zoom)
        self.max_zoom = ensure_tuple(max_zoom)
        if len(self.min_zoom) != len(self.max_zoom):
            raise AssertionError("min_zoom and max_zoom must have same length.")
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.keep_size = keep_size
        self.kwargs = kwargs

        self._zoom: Sequence[float] = [1.0]

    def randomize(self, img: NdarrayOrTensor) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._zoom = [self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom)]
        if len(self._zoom) == 1:
            # to keep the spatial shape ratio, use same random zoom factor for all dims
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 1)
        elif len(self._zoom) == 2 and img.ndim > 3:
            # if 2 zoom factors provided for 3D data, use the first factor for H and W dims, second factor for D dim
            self._zoom = ensure_tuple_rep(self._zoom[0], img.ndim - 2) + ensure_tuple(self._zoom[-1])

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        padding_mode: Optional[Union[NumpyPadMode, PytorchPadMode, str]] = None,
        align_corners: Optional[bool] = None,
        randomize: bool = True,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
                ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
                available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
                One of the listed string values or a user supplied function. Defaults to ``"constant"``.
                The mode to pad data after zooming.
                See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
                https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            align_corners: This only has an effect when mode is
                'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            randomize: whether to execute `randomize()` function first, default to True.

        """
        # match the spatial image dim
        if randomize:
            self.randomize(img=img)

        if not self._do_transform:
            return img

        return Zoom(
            self._zoom,
            keep_size=self.keep_size,
            mode=look_up_option(mode or self.mode, InterpolateMode),
            padding_mode=padding_mode or self.padding_mode,
            align_corners=self.align_corners if align_corners is None else align_corners,
            **self.kwargs,
        )(img)


class AffineGrid(Transform):
    """
    Affine transforms on the coordinates.

    Args:
        rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
            Defaults to no rotation.
        shear_params: shearing factors for affine matrix, take a 3D affine as example::

            [
                [1.0, params[0], params[1], 0.0],
                [params[2], 1.0, params[3], 0.0],
                [params[4], params[5], 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

            a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
        translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
            pixel/voxel relative to the center of the input image. Defaults to no translation.
        scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
            a tuple of 3 floats for 3D. Defaults to `1.0`.
        dtype: data type for the grid computation. Defaults to ``np.float32``.
            If ``None``, use the data type of input data (if `grid` is provided).
        device: device on which the tensor will be allocated, if a new grid is generated.
        affine: If applied, ignore the params (`rotate_params`, etc.) and use the
            supplied matrix. Should be square with each side = num of image spatial
            dimensions + 1.

    .. deprecated:: 0.6.0
        ``as_tensor_output`` is deprecated.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
        dtype: DtypeLike = np.float32,
        affine: Optional[NdarrayOrTensor] = None,
    ) -> None:
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params
        self.device = device
        self.dtype = dtype
        self.affine = affine

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[NdarrayOrTensor] = None
    ) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
        """
        The grid can be initialized with a `spatial_size` parameter, or provided directly as `grid`.
        Therefore, either `spatial_size` or `grid` must be provided.
        When initialising from `spatial_size`, the backend "torch" will be used.

        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Raises:
            ValueError: When ``grid=None`` and ``spatial_size=None``. Incompatible values.

        """
        if grid is None:  # create grid from spatial_size
            if spatial_size is None:
                raise ValueError("Incompatible values: grid=None and spatial_size=None.")
            grid = create_grid(spatial_size, device=self.device, backend="torch", dtype=self.dtype)
        _b = TransformBackends.TORCH if isinstance(grid, torch.Tensor) else TransformBackends.NUMPY
        _device = grid.device if isinstance(grid, torch.Tensor) else self.device
        affine: NdarrayOrTensor
        if self.affine is None:
            spatial_dims = len(grid.shape) - 1
            affine = (
                torch.eye(spatial_dims + 1, device=_device)
                if _b == TransformBackends.TORCH
                else np.eye(spatial_dims + 1)
            )
            if self.rotate_params:
                affine = affine @ create_rotate(spatial_dims, self.rotate_params, device=_device, backend=_b)
            if self.shear_params:
                affine = affine @ create_shear(spatial_dims, self.shear_params, device=_device, backend=_b)
            if self.translate_params:
                affine = affine @ create_translate(spatial_dims, self.translate_params, device=_device, backend=_b)
            if self.scale_params:
                affine = affine @ create_scale(spatial_dims, self.scale_params, device=_device, backend=_b)
        else:
            affine = self.affine

        grid, *_ = convert_data_type(grid, torch.Tensor, device=_device, dtype=self.dtype or grid.dtype)
        affine, *_ = convert_to_dst_type(affine, grid)

        grid = (affine @ grid.reshape((grid.shape[0], -1))).reshape([-1] + list(grid.shape[1:]))
        return grid, affine


class RandAffineGrid(Randomizable, Transform):
    """
    Generate randomised affine grid.

    """

    backend = AffineGrid.backend

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxels to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            device: device to store the output grid data.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: Optional[List[float]] = None
        self.shear_params: Optional[List[float]] = None
        self.translate_params: Optional[List[float]] = None
        self.scale_params: Optional[List[float]] = None

        self.device = device
        self.affine: Optional[NdarrayOrTensor] = None

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Optional[Any] = None) -> None:
        self.rotate_params = self._get_rand_param(self.rotate_range)
        self.shear_params = self._get_rand_param(self.shear_range)
        self.translate_params = self._get_rand_param(self.translate_range)
        self.scale_params = self._get_rand_param(self.scale_range, 1.0)

    def __call__(
        self, spatial_size: Optional[Sequence[int]] = None, grid: Optional[NdarrayOrTensor] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            spatial_size: output grid size.
            grid: grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.

        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        self.randomize()
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            device=self.device,
        )
        _grid: NdarrayOrTensor
        _grid, self.affine = affine_grid(spatial_size, grid)
        return _grid

    def get_transformation_matrix(self) -> Optional[NdarrayOrTensor]:
        """Get the most recently applied transformation matrix"""
        return self.affine


class RandDeformGrid(Randomizable, Transform):
    """
    Generate random deformation grid.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        magnitude_range: Tuple[float, float],
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            spacing: spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range: the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device: device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range

        self.rand_mag = 1.0
        self.as_tensor_output = as_tensor_output
        self.random_offset: np.ndarray
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        self.random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size))).astype(np.float32, copy=False)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]):
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing, device=self.device, backend="torch")
        self.randomize(control_grid.shape[1:])
        _offset, *_ = convert_to_dst_type(self.rand_mag * self.random_offset, control_grid)
        control_grid[: len(spatial_size)] += _offset
        if not self.as_tensor_output:
            control_grid, *_ = convert_data_type(control_grid, output_type=np.ndarray, dtype=np.float32)
        return control_grid


class Resample(Transform):

    backend = [TransformBackends.TORCH]

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        as_tensor_output: bool = True,
        norm_coords: bool = True,
        device: Optional[torch.device] = None,
        dtype: DtypeLike = np.float64,
    ) -> None:
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            norm_coords: whether to normalize the coordinates from `[-(size-1)/2, (size-1)/2]` to
                `[0, size - 1]` (for ``monai/csrc`` implementation) or
                `[-1, 1]` (for torch ``grid_sample`` implementation) to be compatible with the underlying
                resampling API.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.norm_coords = norm_coords
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        grid: Optional[NdarrayOrTensor] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        dtype: DtypeLike = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
                if ``norm_coords`` is True, the grid values must be in `[-(size-1)/2, (size-1)/2]`.
                if ``USE_COMPILED=True`` and ``norm_coords=False``, grid values must be in `[0, size-1]`.
                if ``USE_COMPILED=False`` and ``norm_coords=False``, grid values must be in `[-1, 1]`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            dtype: data type for resampling computation. Defaults to ``self.dtype``.
                To be compatible with other modules, the output data type is always `float32`.

        See also:
            :py:const:`monai.config.USE_COMPILED`
        """
        if grid is None:
            raise ValueError("Unknown grid.")
        _device = img.device if isinstance(img, torch.Tensor) else self.device
        _dtype = dtype or self.dtype or img.dtype
        img_t, *_ = convert_data_type(img, torch.Tensor, device=_device, dtype=_dtype)
        grid_t = convert_to_dst_type(grid, img_t)[0]
        if grid_t is grid:  # copy if needed (convert_data_type converts to contiguous)
            grid_t = grid_t.clone(memory_format=torch.contiguous_format)
        sr = min(len(img_t.shape[1:]), 3)

        if USE_COMPILED:
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                    grid_t[i] = (max(dim, 2) / 2.0 - 0.5 + grid_t[i]) / grid_t[-1:]
            grid_t = moveaxis(grid_t[:sr], 0, -1)  # type: ignore
            _padding_mode = self.padding_mode if padding_mode is None else padding_mode
            _padding_mode = _padding_mode.value if isinstance(_padding_mode, GridSamplePadMode) else _padding_mode
            bound = 1 if _padding_mode == "reflection" else _padding_mode
            _interp_mode = self.mode if mode is None else mode
            _interp_mode = _interp_mode.value if isinstance(_interp_mode, GridSampleMode) else _interp_mode
            if _interp_mode == "bicubic":
                interp = 3
            elif _interp_mode == "bilinear":
                interp = 1
            else:
                interp = _interp_mode  # type: ignore
            out = grid_pull(
                img_t.unsqueeze(0), grid_t.unsqueeze(0), bound=bound, extrapolate=True, interpolation=interp
            )[0]
        else:
            if self.norm_coords:
                for i, dim in enumerate(img_t.shape[1 : 1 + sr]):
                    grid_t[i] = 2.0 / (max(2, dim) - 1.0) * grid_t[i] / grid_t[-1:]
            index_ordering: List[int] = list(range(sr - 1, -1, -1))
            grid_t = moveaxis(grid_t[index_ordering], 0, -1)  # type: ignore
            out = torch.nn.functional.grid_sample(
                img_t.unsqueeze(0),
                grid_t.unsqueeze(0),
                mode=self.mode.value if mode is None else GridSampleMode(mode).value,
                padding_mode=self.padding_mode.value if padding_mode is None else GridSamplePadMode(padding_mode).value,
                align_corners=True,
            )[0]
        out_val, *_ = convert_to_dst_type(out, dst=img, dtype=np.float32)
        return out_val


class Affine(Transform):
    """
    Transform ``img`` given the affine parameters.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = list(set(AffineGrid.backend) & set(Resample.backend))

    @deprecated_arg(name="as_tensor_output", since="0.6")
    @deprecated_arg(name="norm_coords", since="0.8")
    def __init__(
        self,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        affine: Optional[NdarrayOrTensor] = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        normalized: bool = False,
        norm_coords: bool = True,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
        dtype: DtypeLike = np.float32,
        image_only: bool = False,
    ) -> None:
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: shearing factors for affine matrix, take a 3D affine as example::

                [
                    [1.0, params[0], params[1], 0.0],
                    [params[2], 1.0, params[3], 0.0],
                    [params[4], params[5], 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]

                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: scale factor for every spatial dims. a tuple of 2 floats for 2D,
                a tuple of 3 floats for 3D. Defaults to `1.0`.
            affine: If applied, ignore the params (`rotate_params`, etc.) and use the
                supplied matrix. Should be square with each side = num of image spatial
                dimensions + 1.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            normalized: indicating whether the provided `affine` is defined to include a normalization
                transform converting the coordinates from `[-(size-1)/2, (size-1)/2]` (defined in ``create_grid``) to
                `[0, size - 1]` or `[-1, 1]` in order to be compatible with the underlying resampling API.
                If `normalized=False`, additional coordinate normalization will be applied before resampling.
                See also: :py:func:`monai.networks.utils.normalize_transform`.
            device: device on which the tensor will be allocated.
            dtype: data type for resampling computation. Defaults to ``np.float32``.
                If ``None``, use the data type of input data. To be compatible with other modules,
                the output data type is always `float32`.
            image_only: if True return only the image volume, otherwise return (image, affine).

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.
        .. deprecated:: 0.8.1
            ``norm_coords`` is deprecated, please use ``normalized`` instead
            (the new flag is a negation, i.e., ``norm_coords == not normalized``).

        """
        self.affine_grid = AffineGrid(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            affine=affine,
            dtype=dtype,
            device=device,
        )
        self.image_only = image_only
        self.resampler = Resample(norm_coords=not normalized, device=device, dtype=dtype)
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)

    def __call__(
        self,
        img: NdarrayOrTensor,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[NdarrayOrTensor, Tuple[NdarrayOrTensor, NdarrayOrTensor]]:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `USE_COMPILED` is `True`, this argument uses
                ``"nearest"``, ``"bilinear"``, ``"bicubic"`` to indicate 0, 1, 3 order interpolations.
                See also: https://docs.monai.io/en/stable/networks.html#grid-pull
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        """
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        grid, affine = self.affine_grid(spatial_size=sp_size)
        ret = self.resampler(img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)

        return ret if self.image_only else (ret, affine)


class RandAffine(RandomizableTransform):
    """
    Random affine transform.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Affine.backend

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D, a tuple of 6 floats for 3D) for affine matrix,
                take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        RandomizableTransform.__init__(self, prob)

        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.cache_grid = cache_grid
        self._cached_grid = self._init_identity_cache()
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)

    def _init_identity_cache(self):
        """
        Create cache of the identity grid if cache_grid=True and spatial_size is known.
        """
        if self.spatial_size is None:
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size, please specify 'spatial_size'."
                )
            return None
        _sp_size = ensure_tuple(self.spatial_size)
        _ndim = len(_sp_size)
        if _sp_size != fall_back_tuple(_sp_size, [1] * _ndim) or _sp_size != fall_back_tuple(_sp_size, [2] * _ndim):
            # dynamic shape because it falls back to different outcomes
            if self.cache_grid:
                warnings.warn(
                    "cache_grid=True is not compatible with the dynamic spatial_size "
                    f"'spatial_size={self.spatial_size}', please specify 'spatial_size'."
                )
            return None
        return create_grid(spatial_size=_sp_size, device=self.rand_affine_grid.device, backend="torch")

    def get_identity_grid(self, spatial_size: Sequence[int]):
        """
        Return a cached or new identity grid depends on the availability.

        Args:
            spatial_size: non-dynamic spatial size
        """
        ndim = len(spatial_size)
        if spatial_size != fall_back_tuple(spatial_size, [1] * ndim) or spatial_size != fall_back_tuple(
            spatial_size, [2] * ndim
        ):
            raise RuntimeError(f"spatial_size should not be dynamic, got {spatial_size}.")
        return (
            create_grid(spatial_size=spatial_size, device=self.rand_affine_grid.device, backend="torch")
            if self._cached_grid is None
            else self._cached_grid
        )

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandAffine":
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: NdarrayOrTensor,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        randomize: bool = True,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]),
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            randomize: whether to execute `randomize()` function first, default to True.

        """
        if randomize:
            self.randomize()

        # if not doing transform and spatial size doesn't change, nothing to do
        # except convert to float and device
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        do_resampling = self._do_transform or (sp_size != ensure_tuple(img.shape[1:]))
        if not do_resampling:
            img, *_ = convert_data_type(img, dtype=torch.float32, device=self.resampler.device)
        grid = self.get_identity_grid(sp_size)
        if self._do_transform:
            grid = self.rand_affine_grid(grid=grid)
        out: NdarrayOrTensor = self.resampler(
            img=img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )
        return out


class Rand2DElastic(RandomizableTransform):
    """
    Random elastic deformation and affine in 2D.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Resample.backend

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        spacing: Union[Tuple[float, float], float],
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            spacing : distance in between the control points.
            magnitude_range: the random offsets will be generated from ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 2 floats for 2D) for affine matrix, take a 2D affine as example::

                    [
                        [1.0, params[0], 0.0],
                        [params[1], 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        RandomizableTransform.__init__(self, prob)
        self.deform_grid = RandDeformGrid(
            spacing=spacing, magnitude_range=magnitude_range, as_tensor_output=True, device=device
        )
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.device = device
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Rand2DElastic":
        self.deform_grid.set_random_state(seed, state)
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, spatial_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.deform_grid.randomize(spatial_size)
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: NdarrayOrTensor,
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        randomize: bool = True,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (num_channels, H, W),
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            randomize: whether to execute `randomize()` function first, default to True.
        """
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(spatial_size=sp_size)

        if self._do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = self.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(  # type: ignore
                recompute_scale_factor=True,
                input=grid.unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            _device = img.device if isinstance(img, torch.Tensor) else self.device
            grid = create_grid(spatial_size=sp_size, device=_device, backend="torch")
        out: NdarrayOrTensor = self.resampler(
            img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )
        return out


class Rand3DElastic(RandomizableTransform):
    """
    Random elastic deformation and affine in 3D.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.

    """

    backend = Resample.backend

    @deprecated_arg(name="as_tensor_output", since="0.6")
    def __init__(
        self,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized elastic transform.
                defaults to 0.1, with 10% chance returns a randomized elastic transform,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            shear_range: shear range with format matching `rotate_range`, it defines the range to randomly select
                shearing factors(a tuple of 6 floats for 3D) for affine matrix, take a 3D affine as example::

                    [
                        [1.0, params[0], params[1], 0.0],
                        [params[2], 1.0, params[3], 0.0],
                        [params[4], params[5], 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]

            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            device: device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.

        .. deprecated:: 0.6.0
            ``as_tensor_output`` is deprecated.

        """
        RandomizableTransform.__init__(self, prob)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode: GridSampleMode = look_up_option(mode, GridSampleMode)
        self.padding_mode: GridSamplePadMode = look_up_option(padding_mode, GridSamplePadMode)
        self.device = device

        self.rand_offset: np.ndarray
        self.magnitude = 1.0
        self.sigma = 1.0

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Rand3DElastic":
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, grid_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32, copy=False)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: NdarrayOrTensor,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        randomize: bool = True,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (num_channels, H, W, D),
            spatial_size: specifying spatial 3D output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            randomize: whether to execute `randomize()` function first, default to True.
        """
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(grid_size=sp_size)

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        grid = create_grid(spatial_size=sp_size, device=_device, backend="torch")
        if self._do_transform:
            if self.rand_offset is None:
                raise RuntimeError("rand_offset is not initialized.")
            gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=_device)
            offset = torch.as_tensor(self.rand_offset, device=_device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        out: NdarrayOrTensor = self.resampler(
            img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )
        return out


class GridDistortion(Transform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        num_cells: Union[Tuple[int], int],
        distort_steps: Sequence[Sequence[float]],
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Grid distortion transform. Refer to:
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py

        Args:
            num_cells: number of grid cells on each dimension.
            distort_steps: This argument is a list of tuples, where each tuple contains the distort steps of the
                corresponding dimensions (in the order of H, W[, D]). The length of each tuple equals to `num_cells + 1`.
                Each value in the tuple represents the distort step of the related cell.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            device: device on which the tensor will be allocated.

        """
        self.resampler = Resample(mode=mode, padding_mode=padding_mode, device=device)
        self.num_cells = num_cells
        self.distort_steps = distort_steps
        self.device = device

    def __call__(
        self,
        img: NdarrayOrTensor,
        distort_steps: Optional[Sequence[Sequence]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            distort_steps: This argument is a list of tuples, where each tuple contains the distort steps of the
                corresponding dimensions (in the order of H, W[, D]). The length of each tuple equals to `num_cells + 1`.
                Each value in the tuple represents the distort step of the related cell.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

        """
        distort_steps = self.distort_steps if distort_steps is None else distort_steps
        if len(img.shape) != len(distort_steps) + 1:
            raise ValueError("the spatial size of `img` does not match with the length of `distort_steps`")

        all_ranges = []
        num_cells = ensure_tuple_rep(self.num_cells, len(img.shape) - 1)
        for dim_idx, dim_size in enumerate(img.shape[1:]):
            dim_distort_steps = distort_steps[dim_idx]
            ranges = torch.zeros(dim_size, dtype=torch.float32)
            cell_size = dim_size // num_cells[dim_idx]
            prev = 0
            for idx in range(num_cells[dim_idx] + 1):
                start = int(idx * cell_size)
                end = start + cell_size
                if end > dim_size:
                    end = dim_size
                    cur = dim_size
                else:
                    cur = prev + cell_size * dim_distort_steps[idx]
                ranges[start:end] = torch.linspace(prev, cur, end - start)
                prev = cur
            ranges = ranges - (dim_size - 1.0) / 2.0
            all_ranges.append(ranges)

        coords = meshgrid_ij(*all_ranges)
        grid = torch.stack([*coords, torch.ones_like(coords[0])])

        return self.resampler(img, grid=grid, mode=mode, padding_mode=padding_mode)  # type: ignore


class RandGridDistortion(RandomizableTransform):

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        num_cells: Union[Tuple[int], int] = 5,
        prob: float = 0.1,
        distort_limit: Union[Tuple[float, float], float] = (-0.03, 0.03),
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.BORDER,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Random grid distortion transform. Refer to:
        https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/transforms.py

        Args:
            num_cells: number of grid cells on each dimension.
            prob: probability of returning a randomized grid distortion transform. Defaults to 0.1.
            distort_limit: range to randomly distort.
                If single number, distort_limit is picked from (-distort_limit, distort_limit).
                Defaults to (-0.03, 0.03).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            device: device on which the tensor will be allocated.

        """
        RandomizableTransform.__init__(self, prob)
        self.num_cells = num_cells
        if isinstance(distort_limit, (int, float)):
            self.distort_limit = (min(-distort_limit, distort_limit), max(-distort_limit, distort_limit))
        else:
            self.distort_limit = (min(distort_limit), max(distort_limit))
        self.distort_steps: Sequence[Sequence[float]] = ((1.0,),)
        self.grid_distortion = GridDistortion(
            num_cells=num_cells, distort_steps=self.distort_steps, mode=mode, padding_mode=padding_mode, device=device
        )

    def randomize(self, spatial_shape: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return
        self.distort_steps = tuple(
            tuple(1.0 + self.R.uniform(low=self.distort_limit[0], high=self.distort_limit[1], size=n_cells + 1))
            for n_cells in ensure_tuple_rep(self.num_cells, len(spatial_shape))
        )

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
        randomize: bool = True,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            randomize: whether to shuffle the random factors using `randomize()`, default to True.
        """
        if randomize:
            self.randomize(img.shape[1:])
        if not self._do_transform:
            return img
        return self.grid_distortion(img, distort_steps=self.distort_steps, mode=mode, padding_mode=padding_mode)


class GridSplit(Transform):
    """
    Split the image into patches based on the provided grid in 2D.

    Args:
        grid: a tuple define the shape of the grid upon which the image is split. Defaults to (2, 2)
        size: a tuple or an integer that defines the output patch sizes.
            If it's an integer, the value will be repeated for each dimension.
            The default is None, where the patch size will be inferred from the grid shape.

    Example:
        Given an image (torch.Tensor or numpy.ndarray) with size of (3, 10, 10) and a grid of (2, 2),
        it will return a Tensor or array with the size of (4, 3, 5, 5).
        Here, if the `size` is provided, the returned shape will be (4, 3, size, size)

    Note: This transform currently support only image with two spatial dimensions.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, grid: Tuple[int, int] = (2, 2), size: Optional[Union[int, Tuple[int, int]]] = None):
        # Grid size
        self.grid = grid

        # Patch size
        self.size = None if size is None else ensure_tuple_rep(size, len(self.grid))

    def __call__(
        self, image: NdarrayOrTensor, size: Optional[Union[int, Tuple[int, int], np.ndarray]] = None
    ) -> List[NdarrayOrTensor]:
        input_size = self.size if size is None else ensure_tuple_rep(size, len(self.grid))

        if self.grid == (1, 1) and input_size is None:
            return [image]

        split_size, steps = self._get_params(image.shape[1:], input_size)
        patches: List[NdarrayOrTensor]
        if isinstance(image, torch.Tensor):
            unfolded_image = (
                image.unfold(1, split_size[0], steps[0])
                .unfold(2, split_size[1], steps[1])
                .flatten(1, 2)
                .transpose(0, 1)
            )
            # Make a list of contiguous patches
            patches = [p.contiguous() for p in unfolded_image]
        elif isinstance(image, np.ndarray):
            x_step, y_step = steps
            c_stride, x_stride, y_stride = image.strides
            n_channels = image.shape[0]
            strided_image = as_strided(
                image,
                shape=(*self.grid, n_channels, split_size[0], split_size[1]),
                strides=(x_stride * x_step, y_stride * y_step, c_stride, x_stride, y_stride),
                writeable=False,
            )
            # Flatten the first two dimensions
            strided_image = strided_image.reshape(-1, *strided_image.shape[2:])
            # Make a list of contiguous patches
            patches = [np.ascontiguousarray(p) for p in strided_image]
        else:
            raise ValueError(f"Input type [{type(image)}] is not supported.")

        return patches

    def _get_params(
        self, image_size: Union[Sequence[int], np.ndarray], size: Optional[Union[Sequence[int], np.ndarray]] = None
    ):
        """
        Calculate the size and step required for splitting the image
        Args:
            The size of the input image
        """
        if size is None:
            # infer each sub-image size from the image size and the grid
            size = tuple(image_size[i] // self.grid[i] for i in range(len(self.grid)))

        if any(size[i] > image_size[i] for i in range(len(self.grid))):
            raise ValueError(f"The image size ({image_size})is smaller than the requested split size ({size})")

        steps = tuple(
            (image_size[i] - size[i]) // (self.grid[i] - 1) if self.grid[i] > 1 else image_size[i]
            for i in range(len(self.grid))
        )

        return size, steps
