"""3D cortical-surface rendering of modal activity and eigenmodes.

Renders a scalar field (an eigenmode, reconstructed activity, or a criticality map)
on a triangle-mesh cortical surface. Prefers the user's ``yabplot`` if available,
then ``nilearn.plotting.plot_surf``, then ``vedo`` (offscreen-capable). All
dependencies are import-guarded so harmonix imports without any 3D stack.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _has(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def plot_surface(vertices, faces, scalars, cmap: str = "coolwarm",
                 title: str = "", backend: str = "auto", offscreen: bool = True,
                 outfile: Optional[str] = None):
    """Render a scalar field on a cortical surface mesh.

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3).
    faces : np.ndarray, shape (F, 3).
    scalars : np.ndarray, shape (V,)
        Per-vertex scalar (eigenmode, activity, or map).
    cmap : str.
    title : str.
    backend : {'auto', 'yabplot', 'nilearn', 'vedo'}.
    offscreen : bool, render without a window (for headless movie export).
    outfile : str, optional path to save a PNG.

    Returns
    -------
    The backend-specific figure/plotter object, or the saved path.
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, int)
    s = np.asarray(scalars, float).ravel()
    if s.shape[0] != V.shape[0]:
        raise ValueError(f"scalars length {s.shape[0]} != n_vertices {V.shape[0]}")

    order = ["yabplot", "nilearn", "vedo"] if backend == "auto" else [backend]
    for b in order:
        if b == "yabplot" and _has("yabplot"):
            import yabplot
            return yabplot.plot_surface(V, F, s, cmap=cmap, title=title)
        if b == "nilearn" and _has("nilearn"):
            from nilearn import plotting
            fig = plotting.plot_surf((V, F), surf_map=s, cmap=cmap, title=title,
                                     colorbar=True)
            if outfile:
                fig.savefig(outfile, dpi=200)
            return fig
        if b == "vedo" and _has("vedo"):
            import vedo
            if offscreen:
                vedo.settings.default_backend = "vtk"
            mesh = vedo.Mesh([V, F]).cmap(cmap, s).add_scalarbar()
            plt = vedo.Plotter(offscreen=offscreen)
            if outfile:
                plt.show(mesh, title, interactive=False).screenshot(outfile)
                return outfile
            return plt.show(mesh, title, interactive=not offscreen)
    raise ImportError(
        "no 3D backend available; install one of yabplot, nilearn, or vedo "
        "('pip install harmonix[viz]')")


def plot_eigenmode(basis, mode: int, vertices, faces, **kwargs):
    """Convenience: render eigenmode ``mode`` of a basis on its surface."""
    return plot_surface(vertices, faces, basis.Phi[:, mode],
                        title=kwargs.pop("title", f"eigenmode {mode}"), **kwargs)
