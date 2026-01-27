#!/usr/bin/env python3
"""
visualize_3d_octant.py - 3D octant visualization for MRI volumes with lesion overlay.

Creates a 3D visualization showing three orthogonal slices (axial, coronal, sagittal)
with a "missing octant" view that reveals internal brain structures. Epilepsy lesions
are highlighted with colored contours.

Two rendering modes are available:
  1. Matplotlib-based (default): Simple 3D slices with lesion contours
  2. PyVista-based (--show-surface): 3D brain surface mesh with cutaway octant

Usage:
    # Matplotlib mode (default)
    python -m src.utils.visualize_3d_octant \
        --dataset /path/to/Dataset210_MRIe_none \
        --subject MRIe_001 \
        --output output.pdf

    # PyVista mode with brain surface
    python -m src.utils.visualize_3d_octant \
        --dataset /path/to/Dataset210_MRIe_none \
        --subject MRIe_001 \
        --show-surface \
        --output output.png
    
    # Our usage
    python -m src.utils.visualize_3d_octant --dataset /media/mpascual/Sandisk2TB/research/jsddpm/data/epilepsy/source/Dataset210_MRIe_none \
    --subject MRIe_004 --output ./outputs/ouctant_surface_MRIe_004.pdf --modality 0 --show-surface --axial 75 --coronal 100 --sagittal 105 \
    --octant="-++" --mesh-color 0.85 --camera-azimuth 130 --camera-elevation 15 --zoom 2.4

Coordinates follow the RAS convention:
    +x → anterior   |  +y → right-lateral   |  +z → cranial
"""

from __future__ import annotations

import argparse
import pathlib
import warnings
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Final, Any

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401 - registers 3D projection

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

try:
    import cv2
    HAS_CV2 = True
except ModuleNotFoundError:
    HAS_CV2 = False

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False

try:
    import pyvista as pv
    from skimage import measure
    HAS_PYVISTA = True
except ModuleNotFoundError:
    HAS_PYVISTA = False
    pv = None

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default lesion color (red-ish for visibility on grayscale)
LESION_COLOR_RGB = (0.93, 0.40, 0.47)  # Soft red (#EE6677)

# Alternative colors for multiple labels (if needed)
LABEL_LUT: Dict[int, Tuple[float, float, float]] = {
    0: (0.0, 0.0, 0.0),      # background (not used)
    1: (0.93, 0.40, 0.47),   # lesion: red (#EE6677)
    2: (0.13, 0.53, 0.20),   # alt: green (#228833)
    3: (0.27, 0.47, 0.67),   # alt: blue (#4477AA)
}


@dataclass(frozen=True)
class SurfaceConfig:
    """Configuration for PyVista surface rendering."""
    iso_level: float = 0.5
    mesh_alpha: float = 1.0
    mesh_color: Tuple[float, float, float] = (0.75, 0.75, 0.75)
    slice_alpha: float = 0.95
    cmap: str = "gray"
    specular: float = 0.3
    specular_power: float = 20.0
    plane_bias: float = 0.01
    smooth_iterations: int = 5
    smooth_lambda: float = 0.5
    lesion_color: Tuple[float, float, float] = LESION_COLOR_RGB
    lesion_alpha: float = 0.9
    # Octant selection: which corner to cut away
    # Format: (invert_x, invert_y, invert_z) - True means cut the negative side
    octant: Tuple[bool, bool, bool] = (False, False, False)
    # Camera control
    camera_azimuth: Optional[float] = None  # Degrees around z-axis
    camera_elevation: Optional[float] = None  # Degrees from horizontal
    zoom: float = 1.0  # Zoom factor (>1 = closer, <1 = farther)


# =============================================================================
# Data I/O
# =============================================================================

def load_volume(path: pathlib.Path) -> np.ndarray:
    """
    Load a 3D volume from NIfTI (.nii/.nii.gz) or NumPy (.npy) file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the volume file.

    Returns
    -------
    np.ndarray
        3D array of shape (nx, ny, nz) with float64 dtype.
    """
    suffix = "".join(path.suffixes).lower()

    if suffix == ".npy":
        return np.load(path).astype(np.float64, copy=False)

    if suffix in {".nii", ".nii.gz", ".gz"}:
        if nib is None:
            raise ImportError("Reading NIfTI requires nibabel: pip install nibabel")
        img = nib.load(str(path))
        # Convert to RAS orientation for consistent coordinates
        img = nib.as_closest_canonical(img)
        return np.asanyarray(img.get_fdata())

    raise ValueError(f"Unsupported volume format: {path.suffix}")


def pad_to_match(vol: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad volume and segmentation to have identical shapes.
    """
    if vol.shape == seg.shape:
        return vol, seg

    target = tuple(max(sv, ss) for sv, ss in zip(vol.shape, seg.shape))

    def _pad(arr: np.ndarray, name: str) -> np.ndarray:
        pad_width = []
        for s, t in zip(arr.shape, target):
            diff = t - s
            if diff < 0:
                raise ValueError(f"{name} is larger than target along an axis")
            pad_width.append((0, diff))
        return np.pad(arr, pad_width, mode="constant", constant_values=0)

    warnings.warn("Volume/segmentation shape mismatch — zero-padding smaller array")
    return _pad(vol, "volume"), _pad(seg, "segmentation")


# =============================================================================
# Brain Mask Computation (for PyVista surface)
# =============================================================================

def compute_brain_mask(volume: np.ndarray, threshold_percentile: float = 5.0) -> np.ndarray:
    """
    Compute a brain/head mask from an MRI volume using intensity thresholding
    and morphological operations.

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume.
    threshold_percentile : float
        Percentile for intensity threshold (default: 5).

    Returns
    -------
    np.ndarray
        Binary mask of the brain/head region.
    """
    if not HAS_SCIPY:
        raise ImportError("Brain mask computation requires scipy: pip install scipy")

    # Threshold at low percentile to get foreground
    finite_vals = volume[np.isfinite(volume)]
    if finite_vals.size == 0:
        return np.ones(volume.shape, dtype=bool)

    threshold = np.percentile(finite_vals, threshold_percentile)
    mask = volume > threshold

    # Morphological operations to clean up
    struct = ndimage.generate_binary_structure(3, 1)

    # Fill holes
    mask = ndimage.binary_fill_holes(mask)

    # Opening to remove small noise
    mask = ndimage.binary_opening(mask, structure=struct, iterations=2)

    # Keep only largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        mask = labeled == largest_label

    # Closing to smooth boundaries
    mask = ndimage.binary_closing(mask, structure=struct, iterations=2)

    return mask.astype(bool)


# =============================================================================
# Mesh Processing (for PyVista surface)
# =============================================================================

def laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 5,
    lam: float = 0.5
) -> np.ndarray:
    """
    Apply Laplacian smoothing to a mesh.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions (N, 3).
    faces : np.ndarray
        Triangle faces (F, 3).
    iterations : int
        Number of smoothing iterations.
    lam : float
        Smoothing factor (0 = no smoothing, 1 = full averaging).

    Returns
    -------
    np.ndarray
        Smoothed vertex positions.
    """
    n_verts = len(vertices)
    verts = vertices.copy()

    # Build adjacency list
    adjacency = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            adjacency[f[i]].add(f[(i + 1) % 3])
            adjacency[f[i]].add(f[(i + 2) % 3])

    # Smooth iteratively
    for _ in range(iterations):
        new_verts = verts.copy()
        for i in range(n_verts):
            neighbors = list(adjacency[i])
            if neighbors:
                centroid = verts[neighbors].mean(axis=0)
                new_verts[i] = (1 - lam) * verts[i] + lam * centroid
        verts = new_verts

    return verts


def faces_to_pyvista(faces_tri: np.ndarray) -> np.ndarray:
    """Convert (F, 3) triangle indices to PyVista faces array [3, i, j, k, ...]."""
    f = np.hstack([
        np.full((faces_tri.shape[0], 1), 3, dtype=np.int64),
        faces_tri.astype(np.int64)
    ])
    return f.ravel()


# =============================================================================
# Slice & Coordinate Helpers
# =============================================================================

def find_lesion_center(segmentation: np.ndarray) -> Tuple[int, int, int]:
    """
    Find the centroid of the lesion (non-zero voxels).
    """
    coords = np.array(np.where(segmentation > 0))
    if coords.size == 0:
        raise ValueError("No lesion found in segmentation mask")

    center = coords.mean(axis=1).astype(int)
    return tuple(center)


def find_optimal_octant_origin(
    volume: np.ndarray,
    segmentation: Optional[np.ndarray] = None
) -> Tuple[int, int, int]:
    """
    Find optimal slice indices for octant visualization.

    If segmentation is provided, centers on the lesion.
    Otherwise, uses the volume center.

    Returns
    -------
    Tuple[int, int, int]
        (k_axial, i_coronal, j_sagittal) indices.
    """
    nx, ny, nz = volume.shape

    if segmentation is not None and np.any(segmentation > 0):
        cx, cy, cz = find_lesion_center(segmentation)
        # Return in the order (z, x, y) for consistency with the visualization
        return (cz, cx, cy)

    # Default: center of volume
    return (nz // 2, nx // 2, ny // 2)


def compute_brain_extent(octant: np.ndarray, threshold: float = 0.0) -> int:
    """
    Compute the extent of brain tissue in the octant for isotropic display.
    """
    if not np.any(octant > threshold):
        return min(octant.shape)

    coords = np.array(np.where(octant > threshold))
    extent_x = coords[0].max() + 1
    extent_y = coords[1].max() + 1
    extent_z = coords[2].max() + 1

    return int(min(extent_x, extent_y, extent_z))


# =============================================================================
# Matplotlib Rendering Helpers
# =============================================================================

def intensity_to_rgba(
    data2d: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    alpha: float
) -> np.ndarray:
    """Convert grayscale 2D data to RGBA with specified colormap and alpha."""
    eps = np.finfo(float).eps
    normed = (data2d - vmin) / (vmax - vmin + eps)
    fc = plt.get_cmap(cmap)(normed)
    fc[..., 3] = alpha
    return fc


def extract_contour_mask(mask2d: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Extract contour (edge) from a binary mask.
    """
    mask_uint8 = mask2d.astype(np.uint8)

    if HAS_CV2:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    elif HAS_SCIPY:
        struct = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask_uint8, structure=struct).astype(np.uint8)
    else:
        # Simple fallback: shift and compare
        eroded = mask_uint8.copy()
        eroded[1:, :] &= mask_uint8[:-1, :]
        eroded[:-1, :] &= mask_uint8[1:, :]
        eroded[:, 1:] &= mask_uint8[:, :-1]
        eroded[:, :-1] &= mask_uint8[:, 1:]

    edge = mask_uint8 - eroded
    return edge.astype(bool)


def overlay_segmentation(
    facecolors: np.ndarray,
    seg_patch: Optional[np.ndarray],
    *,
    seg_alpha: float = 0.7,
    lut: Dict[int, Tuple[float, float, float]] = LABEL_LUT,
    contour_only: bool = True,
    contour_width: int = 3
) -> np.ndarray:
    """
    Overlay segmentation on RGBA facecolors.
    """
    if seg_patch is None:
        return facecolors

    out = facecolors.copy()

    for label, rgb in lut.items():
        if label == 0:
            continue

        mask = seg_patch == label
        if not mask.any():
            continue

        if contour_only:
            mask = extract_contour_mask(mask, kernel_size=contour_width)

        # Alpha blend
        out[mask, :3] = (
            (1.0 - seg_alpha) * out[mask, :3] +
            seg_alpha * np.asarray(rgb, dtype=float)
        )

    return out


def plot_surface_patch(
    ax: plt.Axes,
    patch: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    alpha: float,
    *,
    seg_patch: Optional[np.ndarray] = None,
    seg_alpha: float = 0.7,
    contour_only: bool = True
) -> None:
    """Plot a single surface patch with optional segmentation overlay."""
    fc = intensity_to_rgba(patch, vmin, vmax, cmap, alpha)
    fc = overlay_segmentation(
        fc, seg_patch,
        seg_alpha=seg_alpha,
        contour_only=contour_only
    )

    # Trim grid arrays to match patch dimensions
    Xf, Yf, Zf = X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]

    ax.plot_surface(
        Xf, Yf, Zf,
        facecolors=fc,
        shade=False,
        rstride=1,
        cstride=1
    )


def draw_wireframe_cube(ax: plt.Axes, dx: int, dy: int, dz: int) -> None:
    """Draw a thin wireframe around the octant cube."""
    edges: Final = [
        ([0, 0, 0], [dx, 0, 0]),
        ([0, 0, 0], [0, dy, 0]),
        ([0, 0, 0], [0, 0, dz]),
        ([dx, dy, 0], [0, dy, 0]),
        ([dx, dy, 0], [dx, 0, 0]),
        ([dx, dy, 0], [dx, dy, dz]),
        ([dx, 0, dz], [0, 0, dz]),
        ([dx, 0, dz], [dx, dy, dz]),
        ([0, dy, dz], [0, 0, dz]),
        ([0, dy, dz], [dx, dy, dz]),
        ([0, 0, dz], [dx, 0, dz]),
        ([0, 0, 0], [0, dy, dz]),
    ]
    for start, end in edges:
        ax.plot3D(*zip(start, end), color="k", linewidth=0.4)


# =============================================================================
# Matplotlib-based Octant Visualization
# =============================================================================

def plot_octant(
    volume: np.ndarray,
    slice_indices: Tuple[int, int, int],
    *,
    segmentation: Optional[np.ndarray] = None,
    cmap: str = "gray",
    alpha: float = 0.95,
    seg_alpha: float = 0.85,
    contour_only: bool = True,
    isotropic: bool = True,
    show_wireframe: bool = False,
    show_axes: bool = False,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    figsize: Tuple[float, float] = (8, 8),
    elev: float = 25,
    azim: float = 45,
    save: Optional[pathlib.Path] = None,
    dpi: int = 300,
    transparent: bool = True
) -> plt.Figure:
    """
    Create 3D octant visualization of MRI volume with lesion overlay (Matplotlib).

    Shows three orthogonal slices (axial, coronal, sagittal) from the specified
    origin point, creating a "missing octant" view into the brain.
    """
    if volume.ndim != 3:
        raise ValueError("`volume` must be 3-D")
    if segmentation is not None and segmentation.shape != volume.shape:
        raise ValueError("`segmentation` shape must match `volume`")

    nx, ny, nz = volume.shape
    k_a, i_c, j_s = slice_indices

    if not (0 <= k_a < nz and 0 <= i_c < nx and 0 <= j_s < ny):
        raise ValueError(f"Slice indices out of bounds: {slice_indices} for shape {volume.shape}")

    # Extract octant
    vol_oct = volume[i_c:, j_s:, k_a:]
    seg_oct = segmentation[i_c:, j_s:, k_a:] if segmentation is not None else None

    dx, dy, dz = vol_oct.shape
    if min(dx, dy, dz) == 0:
        raise ValueError("Chosen voxel lies on boundary — octant is empty")

    # Optionally crop to isotropic cube
    if isotropic:
        L = compute_brain_extent(vol_oct, threshold=volume.min() + 0.1 * (volume.max() - volume.min()))
        L = min(L, dx, dy, dz)
        dx = dy = dz = L

    vmin, vmax = float(volume.min()), float(volume.max())

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((dx, dy, dz))

    # Axial slice (XY @ Z=0)
    patch_ax = vol_oct[:dx, :dy, 0]
    seg_ax = seg_oct[:dx, :dy, 0] if seg_oct is not None else None

    x_edges = np.arange(dx + 1)
    y_edges = np.arange(dy + 1)
    X_ax, Y_ax = np.meshgrid(x_edges, y_edges, indexing="ij")
    Z_ax = np.zeros_like(X_ax)

    plot_surface_patch(
        ax, patch_ax, X_ax, Y_ax, Z_ax,
        vmin, vmax, cmap, alpha,
        seg_patch=seg_ax, seg_alpha=seg_alpha, contour_only=contour_only
    )

    # Coronal slice (YZ @ X=0)
    patch_co = vol_oct[0, :dy, :dz]
    seg_co = seg_oct[0, :dy, :dz] if seg_oct is not None else None

    y_edges = np.arange(dy + 1)
    z_edges = np.arange(dz + 1)
    Y_co, Z_co = np.meshgrid(y_edges, z_edges, indexing="ij")
    X_co = np.zeros_like(Y_co)

    plot_surface_patch(
        ax, patch_co, X_co, Y_co, Z_co,
        vmin, vmax, cmap, alpha,
        seg_patch=seg_co, seg_alpha=seg_alpha, contour_only=contour_only
    )

    # Sagittal slice (XZ @ Y=0)
    patch_sa = vol_oct[:dx, 0, :dz]
    seg_sa = seg_oct[:dx, 0, :dz] if seg_oct is not None else None

    x_edges = np.arange(dx + 1)
    z_edges = np.arange(dz + 1)
    X_sa, Z_sa = np.meshgrid(x_edges, z_edges, indexing="ij")
    Y_sa = np.zeros_like(X_sa)

    plot_surface_patch(
        ax, patch_sa, X_sa, Y_sa, Z_sa,
        vmin, vmax, cmap, alpha,
        seg_patch=seg_sa, seg_alpha=seg_alpha, contour_only=contour_only
    )

    # Styling
    if show_wireframe:
        draw_wireframe_cube(ax, dx, dy, dz)

    if show_axes:
        ax.set_xlabel(xlabel or "Anterior (+x)")
        ax.set_ylabel(ylabel or "Right (+y)")
        ax.set_zlabel(zlabel or "Cranial (+z)")
    else:
        ax.set_axis_off()

    ax.set_xlim(0, dx)
    ax.set_ylim(0, dy)
    ax.set_zlim(0, dz)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)

    fig.tight_layout(pad=0)

    if save is not None:
        fig.savefig(
            save, dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=transparent
        )
        print(f"Saved: {save}")

    return fig


# =============================================================================
# PyVista-based Octant Visualization with Brain Surface
# =============================================================================

def plot_octant_with_surface(
    volume: np.ndarray,
    slice_indices: Tuple[int, int, int],
    *,
    segmentation: Optional[np.ndarray] = None,
    cfg: SurfaceConfig = SurfaceConfig(),
    window_size: Tuple[int, int] = (1200, 1000),
    save: Optional[pathlib.Path] = None,
    off_screen: bool = True,
) -> Any:
    """
    Create 3D octant visualization with brain surface mesh using PyVista.

    Renders a 3D brain surface with a cutaway octant revealing three orthogonal
    slices inside. Optionally highlights lesions on the surface.

    Parameters
    ----------
    volume : np.ndarray
        3D MRI volume (nx, ny, nz).
    slice_indices : Tuple[int, int, int]
        (k_axial, i_coronal, j_sagittal) — origin voxel for the octant.
    segmentation : np.ndarray, optional
        Lesion segmentation mask.
    cfg : SurfaceConfig
        Rendering configuration.
    window_size : Tuple[int, int]
        Render window dimensions.
    save : pathlib.Path, optional
        Output file path.
    off_screen : bool
        If True, render off-screen (for saving).

    Returns
    -------
    pv.Plotter
        The PyVista plotter object.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista surface rendering requires: pip install pyvista scikit-image")

    nx, ny, nz = volume.shape
    k_a, i_c, j_s = slice_indices

    # Clamp indices
    margin = 1
    k = int(np.clip(k_a, margin, nz - 1 - margin))
    i = int(np.clip(i_c, margin, nx - 1 - margin))
    j = int(np.clip(j_s, margin, ny - 1 - margin))

    logger.info(f"[surface] Volume shape: {volume.shape}, octant origin: (i={i}, j={j}, k={k})")

    # Compute brain mask
    mask = compute_brain_mask(volume)
    logger.info(f"[surface] Brain mask: {mask.sum()} voxels ({100*mask.sum()/mask.size:.1f}%)")

    # Mask the volume (NaN outside brain for transparent rendering)
    vol_masked = volume.copy().astype(np.float32)
    vol_masked[~mask] = np.nan

    # Compute intensity range from brain region
    inside = vol_masked[np.isfinite(vol_masked)]
    if inside.size > 0:
        vmin, vmax = np.percentile(inside, [1.0, 99.0])
    else:
        vmin, vmax = float(np.nanmin(vol_masked)), float(np.nanmax(vol_masked))

    # Create plotter
    pv.global_theme.background = "white"
    plotter = pv.Plotter(off_screen=off_screen or (save is not None), window_size=list(window_size))

    try:
        plotter.enable_anti_aliasing("msaa")
    except Exception:
        pass

    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass

    # ─────────────────────── Brain Surface Mesh ───────────────────────
    # Create surface using marching cubes
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=cfg.iso_level)
    logger.info(f"[surface] Marching cubes: {len(verts)} vertices, {len(faces)} faces")

    # Smooth the mesh
    verts = laplacian_smooth(verts, faces, iterations=cfg.smooth_iterations, lam=cfg.smooth_lambda)

    # Create PyVista mesh
    mesh_full = pv.PolyData(verts, faces_to_pyvista(faces))

    # Clip away the octant based on configuration
    # cfg.octant = (invert_x, invert_y, invert_z)
    # When False: cut x>=i (positive side), When True: cut x<=i (negative side)
    inv_x, inv_y, inv_z = cfg.octant

    if inv_x:
        bounds_x = (-0.5, i + 0.5)
    else:
        bounds_x = (i - 0.5, nx - 0.5)

    if inv_y:
        bounds_y = (-0.5, j + 0.5)
    else:
        bounds_y = (j - 0.5, ny - 0.5)

    if inv_z:
        bounds_z = (-0.5, k + 0.5)
    else:
        bounds_z = (k - 0.5, nz - 0.5)

    bounds = (bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1], bounds_z[0], bounds_z[1])
    mesh_clip = mesh_full.clip_box(bounds=bounds, invert=True, merge_points=True)
    logger.info(f"[surface] Clipped mesh: {mesh_clip.n_points} vertices (octant: inv_x={inv_x}, inv_y={inv_y}, inv_z={inv_z})")

    # Add the surface mesh
    plotter.add_mesh(
        mesh_clip,
        color=cfg.mesh_color,
        opacity=cfg.mesh_alpha,
        smooth_shading=True,
        specular=float(np.clip(cfg.specular, 0.0, 1.0)),
        specular_power=float(cfg.specular_power),
        show_edges=False,
    )

    # ─────────────────────── Lesion Surface (if provided) ───────────────────────
    if segmentation is not None and np.any(segmentation > 0):
        try:
            lesion_verts, lesion_faces, _, _ = measure.marching_cubes(
                segmentation.astype(np.float32), level=0.5
            )
            lesion_mesh = pv.PolyData(lesion_verts, faces_to_pyvista(lesion_faces))

            # Clip lesion to visible octant region (same clipping as brain)
            lesion_clipped = lesion_mesh.clip_box(bounds=bounds, invert=True, merge_points=True)

            # Also show the full lesion (unclipped) if it's outside the cut octant
            if lesion_clipped.n_points > 0:
                plotter.add_mesh(
                    lesion_clipped,
                    color=cfg.lesion_color,
                    opacity=cfg.lesion_alpha,
                    smooth_shading=True,
                    show_edges=False,
                )
                logger.info(f"[surface] Lesion mesh (clipped) added: {lesion_clipped.n_points} vertices")

            # Add unclipped lesion with slight transparency for visibility
            plotter.add_mesh(
                lesion_mesh,
                color=cfg.lesion_color,
                opacity=cfg.lesion_alpha * 0.7,
                smooth_shading=True,
                show_edges=False,
            )
            logger.info(f"[surface] Full lesion mesh added: {lesion_mesh.n_points} vertices")
        except Exception as e:
            logger.warning(f"[surface] Could not create lesion surface: {e}")

    # ─────────────────────── Slice Surfaces ───────────────────────
    # Create uniform grid for slicing
    grid = pv.ImageData()
    grid.dimensions = np.array(vol_masked.shape, dtype=int) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (-0.5, -0.5, -0.5)
    grid.cell_data["I"] = vol_masked.ravel(order="F")
    grid = grid.cell_data_to_point_data(pass_cell_data=True)

    # Determine clip directions based on octant configuration
    # When inv_x=False (cut positive), we clip to keep x>=i, so normal=(1,0,0), invert=False
    # When inv_x=True (cut negative), we clip to keep x<=i, so normal=(-1,0,0), invert=False
    clip_x_normal = (-1, 0, 0) if inv_x else (1, 0, 0)
    clip_y_normal = (0, -1, 0) if inv_y else (0, 1, 0)
    clip_z_normal = (0, 0, -1) if inv_z else (0, 0, 1)

    # Axial slice (XY @ z=k)
    z0 = float(k) + (cfg.plane_bias if not inv_z else -cfg.plane_bias)
    slc = grid.slice(normal=(0, 0, 1), origin=(0.0, 0.0, z0))
    slc = slc.clip(normal=clip_x_normal, origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=clip_y_normal, origin=(0.0, float(j), 0.0), invert=False)
    plotter.add_mesh(
        slc, scalars="I", cmap=cfg.cmap, clim=(vmin, vmax),
        opacity=cfg.slice_alpha, nan_opacity=0.0, show_scalar_bar=False,
    )

    # Coronal slice (YZ @ x=i)
    x0 = float(i) + (cfg.plane_bias if not inv_x else -cfg.plane_bias)
    slc = grid.slice(normal=(1, 0, 0), origin=(x0, 0.0, 0.0))
    slc = slc.clip(normal=clip_y_normal, origin=(0.0, float(j), 0.0), invert=False)
    slc = slc.clip(normal=clip_z_normal, origin=(0.0, 0.0, float(k)), invert=False)
    plotter.add_mesh(
        slc, scalars="I", cmap=cfg.cmap, clim=(vmin, vmax),
        opacity=cfg.slice_alpha, nan_opacity=0.0, show_scalar_bar=False,
    )

    # Sagittal slice (XZ @ y=j)
    y0 = float(j) + (cfg.plane_bias if not inv_y else -cfg.plane_bias)
    slc = grid.slice(normal=(0, 1, 0), origin=(0.0, y0, 0.0))
    slc = slc.clip(normal=clip_x_normal, origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=clip_z_normal, origin=(0.0, 0.0, float(k)), invert=False)
    plotter.add_mesh(
        slc, scalars="I", cmap=cfg.cmap, clim=(vmin, vmax),
        opacity=cfg.slice_alpha, nan_opacity=0.0, show_scalar_bar=False,
    )

    # ─────────────────────── Camera Setup ───────────────────────
    if mask.any():
        idx = np.argwhere(mask)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = idx.min(0), idx.max(0)
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
        brain_size = max(xmax - xmin, ymax - ymin, zmax - zmin)
    else:
        center = (nx / 2, ny / 2, nz / 2)
        brain_size = max(nx, ny, nz)

    plotter.set_focus(center)

    # Camera position based on octant and user parameters
    # Use tighter distance based on actual brain size, not full volume
    dist = 1.8 * brain_size

    if cfg.camera_azimuth is not None and cfg.camera_elevation is not None:
        # User-specified camera angles
        azim_rad = np.radians(cfg.camera_azimuth)
        elev_rad = np.radians(cfg.camera_elevation)
        cam_x = center[0] + dist * np.cos(elev_rad) * np.cos(azim_rad)
        cam_y = center[1] + dist * np.cos(elev_rad) * np.sin(azim_rad)
        cam_z = center[2] + dist * np.sin(elev_rad)
        cam_pos = (cam_x, cam_y, cam_z)
    else:
        # Auto-position based on octant selection (view the cut corner)
        # Position camera to look at the open octant
        x_sign = -1 if inv_x else 1
        y_sign = -1 if inv_y else 1
        z_sign = 0.7 if inv_z else 0.9  # slightly lower for bottom octants

        cam_pos = (
            center[0] + x_sign * dist * 0.7,
            center[1] + y_sign * dist * 0.7,
            center[2] + z_sign * dist * 0.5
        )

    plotter.set_position(cam_pos)
    plotter.set_viewup((0, 0, 1))

    # Tight framing: use larger view angle to fill the frame
    plotter.camera.SetViewAngle(35)
    plotter.reset_camera_clipping_range()

    # Apply zoom factor
    if cfg.zoom != 1.0:
        plotter.camera.zoom(cfg.zoom)

    logger.info(f"[surface] Camera position: {cam_pos}, focus: {center}, zoom: {cfg.zoom}")

    # Save or show
    if save is not None:
        suffix = save.suffix.lower()
        if suffix == ".pdf":
            plotter.save_graphic(str(save))
        else:
            plotter.screenshot(str(save))
        print(f"Saved: {save}")

    return plotter


# =============================================================================
# CLI Interface
# =============================================================================

def create_legend(
    labels: Dict[str, Tuple[float, float, float]],
    save: Optional[pathlib.Path] = None,
    figsize: Tuple[float, float] = (4, 0.6)
) -> plt.Figure:
    """Create a standalone legend for the segmentation colors."""
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(facecolor=rgb, edgecolor="black", label=name)
        for name, rgb in labels.items()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        handlelength=1.5,
        fontsize=12
    )
    ax.axis("off")

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved legend: {save}")

    return fig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="3D octant visualization of MRI volume with lesion overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_argument_group("Input (choose one method)")
    input_group.add_argument(
        "--image", type=pathlib.Path,
        help="Path to image volume (.nii.gz or .npy)"
    )
    input_group.add_argument(
        "--mask", type=pathlib.Path,
        help="Path to segmentation mask (.nii.gz or .npy)"
    )
    input_group.add_argument(
        "--dataset", type=pathlib.Path,
        help="Path to dataset directory (e.g., Dataset210_MRIe_none)"
    )
    input_group.add_argument(
        "--subject", type=str,
        help="Subject ID (e.g., MRIe_001) when using --dataset"
    )
    input_group.add_argument(
        "--modality", type=int, default=0, choices=[0, 1],
        help="Modality index: 0=FLAIR, 1=T1 (default: 0)"
    )

    # Slice selection
    slice_group = parser.add_argument_group("Slice selection")
    slice_group.add_argument(
        "--axial", "-z", type=int, default=None,
        help="Axial slice index (auto-detect if not specified)"
    )
    slice_group.add_argument(
        "--coronal", "-x", type=int, default=None,
        help="Coronal slice index (auto-detect if not specified)"
    )
    slice_group.add_argument(
        "--sagittal", "-y", type=int, default=None,
        help="Sagittal slice index (auto-detect if not specified)"
    )

    # Visualization options
    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument(
        "--show-surface", action="store_true",
        help="Use PyVista to render 3D brain surface with cutaway octant"
    )
    vis_group.add_argument(
        "--cmap", default="gray",
        help="Colormap for image (default: gray)"
    )
    vis_group.add_argument(
        "--alpha", type=float, default=0.95,
        help="Image transparency (default: 0.95)"
    )
    vis_group.add_argument(
        "--seg-alpha", type=float, default=0.85,
        help="Segmentation overlay opacity (default: 0.85)"
    )
    vis_group.add_argument(
        "--fill", action="store_true",
        help="Fill lesion regions instead of showing contours only (matplotlib mode)"
    )
    vis_group.add_argument(
        "--elev", type=float, default=25,
        help="Viewing elevation angle (default: 25, matplotlib mode)"
    )
    vis_group.add_argument(
        "--azim", type=float, default=45,
        help="Viewing azimuth angle (default: 45, matplotlib mode)"
    )
    vis_group.add_argument(
        "--wireframe", action="store_true",
        help="Show bounding box wireframe (matplotlib mode)"
    )
    vis_group.add_argument(
        "--show-axes", action="store_true",
        help="Show axis labels and ticks"
    )
    vis_group.add_argument(
        "--no-isotropic", action="store_true",
        help="Don't crop to isotropic cube (matplotlib mode)"
    )
    vis_group.add_argument(
        "--mesh-color", type=str, default=None,
        help="Brain surface color as hex (e.g., '#888888') or gray float 0-1 (PyVista mode)"
    )
    vis_group.add_argument(
        "--mesh-alpha", type=float, default=1.0,
        help="Brain surface opacity (default: 1.0, PyVista mode)"
    )
    vis_group.add_argument(
        "--octant", type=str, default="+++",
        choices=["+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---"],
        help="Which octant to cut away (PyVista mode). Format: XYZ where + means cut positive side, "
             "- means cut negative side. Examples: '+++' (default, cuts right-anterior-superior), "
             "'---' (cuts left-posterior-inferior). Use '-++' to see left-side lesions."
    )
    vis_group.add_argument(
        "--camera-azimuth", type=float, default=None,
        help="Camera azimuth angle in degrees (PyVista mode). 0=+X, 90=+Y, etc."
    )
    vis_group.add_argument(
        "--camera-elevation", type=float, default=None,
        help="Camera elevation angle in degrees (PyVista mode). 0=horizontal, 90=top-down"
    )
    vis_group.add_argument(
        "--zoom", type=float, default=1.4,
        help="Zoom factor (PyVista mode). >1 = closer/tighter, <1 = farther. Default: 1.4"
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output", "-o", type=pathlib.Path,
        help="Output file path (shows interactively if not specified)"
    )
    output_group.add_argument(
        "--dpi", type=int, default=300,
        help="Output resolution (default: 300, matplotlib mode)"
    )
    output_group.add_argument(
        "--figsize", type=float, nargs=2, default=[8, 8],
        help="Figure size in inches (default: 8 8, matplotlib mode)"
    )
    output_group.add_argument(
        "--window-size", type=int, nargs=2, default=[1200, 1000],
        help="Window size in pixels (default: 1200 1000, PyVista mode)"
    )
    output_group.add_argument(
        "--legend", type=pathlib.Path,
        help="Also save a standalone legend to this path"
    )
    output_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Resolve input paths
    if args.dataset and args.subject:
        dataset_path = args.dataset
        subject = args.subject
        modality = args.modality

        image_path = dataset_path / "imagesTr" / f"{subject}_{modality:04d}.nii.gz"
        mask_path = dataset_path / "labelsTr" / f"{subject}.nii.gz"

        if not image_path.exists():
            # Try imagesTs
            image_path = dataset_path / "imagesTs" / f"{subject}_{modality:04d}.nii.gz"
            mask_path = dataset_path / "labelsTs" / f"{subject}.nii.gz"

        if not image_path.exists():
            raise FileNotFoundError(f"Could not find image for subject {subject}")

    elif args.image:
        image_path = args.image
        mask_path = args.mask
    else:
        raise ValueError("Must specify either --image or --dataset with --subject")

    print(f"Loading image: {image_path}")
    volume = load_volume(image_path)
    print(f"  Shape: {volume.shape}")

    segmentation = None
    if mask_path and mask_path.exists():
        print(f"Loading mask: {mask_path}")
        segmentation = load_volume(mask_path).astype(np.int32)
        volume, segmentation = pad_to_match(volume, segmentation)

        n_lesion = np.sum(segmentation > 0)
        print(f"  Lesion voxels: {n_lesion}")

    # Determine slice indices
    if all(idx is not None for idx in [args.axial, args.coronal, args.sagittal]):
        slice_indices = (args.axial, args.coronal, args.sagittal)
    else:
        slice_indices = find_optimal_octant_origin(volume, segmentation)
        print(f"Auto-detected slice indices: (z={slice_indices[0]}, x={slice_indices[1]}, y={slice_indices[2]})")

    # Choose rendering mode
    if args.show_surface:
        if not HAS_PYVISTA:
            raise ImportError(
                "PyVista surface rendering requires: pip install pyvista scikit-image scipy"
            )

        # Build surface config
        mesh_color = SurfaceConfig.mesh_color
        if args.mesh_color is not None:
            if args.mesh_color.startswith("#"):
                import matplotlib.colors as mcolors
                mesh_color = mcolors.to_rgb(args.mesh_color)
            else:
                g = float(args.mesh_color)
                mesh_color = (g, g, g)

        # Parse octant string (e.g., "+++" -> (False, False, False), "-++" -> (True, False, False))
        octant_str = args.octant
        octant = (octant_str[0] == '-', octant_str[1] == '-', octant_str[2] == '-')
        logger.info(f"Octant selection: {octant_str} -> invert (x={octant[0]}, y={octant[1]}, z={octant[2]})")

        cfg = SurfaceConfig(
            cmap=args.cmap,
            slice_alpha=args.alpha,
            mesh_color=mesh_color,
            mesh_alpha=args.mesh_alpha,
            lesion_alpha=args.seg_alpha,
            octant=octant,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            zoom=args.zoom,
        )

        plotter = plot_octant_with_surface(
            volume,
            slice_indices,
            segmentation=segmentation,
            cfg=cfg,
            window_size=tuple(args.window_size),
            save=args.output,
            off_screen=args.output is not None,
        )

        if args.output is None:
            plotter.show()

    else:
        # Matplotlib mode
        fig = plot_octant(
            volume,
            slice_indices,
            segmentation=segmentation,
            cmap=args.cmap,
            alpha=args.alpha,
            seg_alpha=args.seg_alpha,
            contour_only=not args.fill,
            isotropic=not args.no_isotropic,
            show_wireframe=args.wireframe,
            show_axes=args.show_axes,
            figsize=tuple(args.figsize),
            elev=args.elev,
            azim=args.azim,
            save=args.output,
            dpi=args.dpi
        )

        if args.output is None:
            plt.show()

        plt.close(fig)

    # Create legend if requested
    if args.legend:
        create_legend(
            {"Epilepsy Lesion (FCD)": LESION_COLOR_RGB},
            save=args.legend
        )


if __name__ == "__main__":
    main()
