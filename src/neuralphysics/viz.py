# I used Claude Code to help implement and package the visualisations.

r"""Figures: the disc, the geometry, the tissue map, and the inversion's progress.

The only module that imports matplotlib. Plotting functions take their scientific
data as arguments and return the figures they build. They use the palettes and
panel limit defined below, but do not call `show` or `close`; display, saving and
cleanup remain the caller's responsibility.

Two things follow from the geometry rather than from taste. Fields on the
cross-section are drawn on the PHYSICAL disc, using the mapped node positions, not
on the reference square: the square is a coordinate system, not the limb, and a
figure drawn on it puts the tissue interfaces in the wrong place. And a depth
slice is specified as a fraction of the axial extent rather than as an index, so
the same call selects the same physical plane at every resolution.

IPython is deliberately not imported. `animate_slices` returns the figure and the
FuncAnimation; rendering it inline is done in the notebook.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
)
from matplotlib.patches import Circle

from .utils import to_numpy

__all__ = [
    "TISSUE_COLOURS",
    "PARAM_STYLE",
    "cell_corners",
    "slice_index",
    "plot_disc",
    "plot_jacobian_fields",
    "plot_metric_fields",
    "plot_tissue_map",
    "plot_convergence",
    "plot_ensemble_misfit",
    "plot_error_vs_noise",
    "plot_table",
    "plot_radius_convergence",
    "plot_field_panels",
    "plot_residual_panels",
    "plot_geometry_panels",
    "animate_slices",
]


# ---- palettes ---------------------------------------------------------------

# One colour per tissue, used by every figure that shows the layering, so that the
# tissue map, the animation and the convergence plots agree on what "fat" looks
# like. A palette is not configuration: nothing about the physics or the run
# changes with it.
TISSUE_COLOURS = {
    "bone": "#dcd7c9",
    "muscle": "#c0504d",
    "fat": "#f2c85b",
    "skin": "#e0a878",
}

# One (colour, linestyle) per recovered parameter, tied to the tissue palette.
# Muscle contributes two parameters, so its transverse and longitudinal components
# share a colour and are separated by linestyle rather than being given two
# unrelated colours.
PARAM_STYLE = {
    "bone": (TISSUE_COLOURS["bone"], "-"),
    "muscle_T": (TISSUE_COLOURS["muscle"], "-"),
    "muscle_L": (TISSUE_COLOURS["muscle"], "--"),
    "fat": (TISSUE_COLOURS["fat"], "-"),
    "skin": (TISSUE_COLOURS["skin"], "-"),
}


# ---- primitives -------------------------------------------------------------


def cell_corners(A):
    """(n, n) node positions -> (n + 1, n + 1) cell corners.

    Odd-reflect padding extrapolates one node beyond each edge, and averaging each
    2x2 block of the padded array estimates the corner between four node-centred
    samples. This supplies the corner-coordinate shape required by
    `pcolormesh(..., shading='flat')` on the curvilinear grid without interpolating
    the cell colours.
    """
    P = np.pad(to_numpy(A), 1, mode="reflect", reflect_type="odd")
    return 0.25 * (P[:-1, :-1] + P[1:, :-1] + P[:-1, 1:] + P[1:, 1:])


def slice_index(n, z_fraction):
    """Node index of the plane at `z_fraction` of the axial extent.

    The fraction is the interface: an index means a different physical plane at
    every resolution, so a figure specified by index cannot be reproduced at
    another n, while one specified by fraction can. Node k sits at z = k L/(n-1),
    so the fraction is scale-free and L never enters.
    """
    if not 0.0 <= z_fraction <= 1.0:
        raise ValueError(f"z_fraction must lie in [0, 1], got {z_fraction}")
    return int(round(z_fraction * (n - 1)))


def plot_disc(ax, field2d, Xd, Yd, vmax, title, cmap="seismic"):
    """Colour an (n, n) cross-section on the physical disc, on a symmetric scale.

    The axis is greyed and the rim is outlined so that the circular cross-section
    stays visible where the field is near zero and the colormap is white; without
    it the disc reads as a square. The rim is traced through the four edges of the
    node array, which are the outer ring of the mapped grid.

    Returns the QuadMesh, so the caller can attach a colorbar or update its data
    for a later frame.
    """
    Xd, Yd = to_numpy(Xd), to_numpy(Yd)
    ax.set_facecolor("0.85")
    pm = ax.pcolormesh(
        Xd,
        Yd,
        to_numpy(field2d),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        shading="gouraud",
    )
    bx = np.concatenate([Xd[0, :], Xd[:, -1], Xd[-1, ::-1], Xd[::-1, 0]])
    by = np.concatenate([Yd[0, :], Yd[:, -1], Yd[-1, ::-1], Yd[::-1, 0]])
    ax.plot(bx, by, color="k", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return pm


def _series(a):
    """A 1-D history as a numpy array, whether it arrives as a list or a tensor.

    Histories are accumulated as plain lists by whatever drives the optimiser, so
    requiring a tensor here would push a conversion onto every call site.
    """
    return np.asarray(to_numpy(a) if hasattr(a, "detach") else a)


def _as_sequence(x):
    """A list, whether one field or several were passed.

    A single source is the common case and a bare tensor is what a caller naturally
    writes for it; requiring a one-element list would be noise at every call site.
    """
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _rim_path(Xd, Yd):
    """(x, y) of the outer node ring, traced once round."""
    Xd, Yd = to_numpy(Xd), to_numpy(Yd)
    return (
        np.concatenate([Xd[0, :], Xd[:, -1], Xd[-1, ::-1], Xd[::-1, 0]]),
        np.concatenate([Yd[0, :], Yd[:, -1], Yd[-1, ::-1], Yd[::-1, 0]]),
    )


def _tissue_cmap(layer_names):
    """(ListedColormap, BoundaryNorm) mapping integer layer labels to colours."""
    cmap = ListedColormap([TISSUE_COLOURS[nm] for nm in layer_names])
    return cmap, BoundaryNorm(np.arange(len(layer_names) + 1) - 0.5, cmap.N)


def _tissue_legend_handles(layer_names):
    return [
        plt.Line2D(
            [],
            [],
            marker="s",
            ls="",
            ms=9,
            mfc=TISSUE_COLOURS[nm],
            mec="0.4",
            label=nm,
        )
        for nm in layer_names
    ]


# ---- geometry ---------------------------------------------------------------


def plot_jacobian_fields(grid):
    """Heatmaps of the four raw Jacobian entries on the reference cube.

    These are the signed first derivatives d(x, y)/d(xi, eta) that the metric is
    built from, and they are not recoverable from it: the metric drops the signs
    and collapses four fields into three symmetric coefficients and a determinant.
    Plotted to check the map itself rather than its consequences.

    The halo ring is stripped. The Jacobian is laid out [eta, xi], the layout
    central differences produce, so it is drawn without a transpose and the
    orientation matches `plot_metric_fields`, which transposes because its fields
    are stored [xi, eta].
    """
    J = grid.jacobian
    fields = [
        (J[0, 0], r"$J_{xx}=\partial x/\partial\xi$"),
        (J[0, 1], r"$J_{xy}=\partial x/\partial\eta$"),
        (J[1, 0], r"$J_{yx}=\partial y/\partial\xi$"),
        (J[1, 1], r"$J_{yy}=\partial y/\partial\eta$"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (fld, title) in zip(axes, fields):
        im = ax.imshow(to_numpy(fld[1:-1, 1:-1]), origin="lower", cmap="jet")
        ax.set_title(title)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        r"raw Jacobian entries $J=\partial(x,y)/\partial(\xi,\eta)$ "
        r"on the reference cube (disc cross-section)"
    )
    fig.tight_layout()
    return fig


def plot_metric_fields(grid):
    """Heatmaps of the four coefficient fields the operator consumes.

    A single z-slice is drawn and the choice of slice is immaterial: the extrusion
    is straight, so the metric is exactly z-invariant. det(J) is shown alongside
    the three transverse coefficients because it is the mesh-validity quantity --
    the gate is det(J) > 0 everywhere -- and it is what the Jacobian plots are
    checked against.
    """
    mxx, mxy, myy, _ = grid.metric
    fields = [
        (grid.det_J, "det(J) = m_L"),
        (mxx, "m_xx"),
        (myy, "m_yy"),
        (mxy, "m_xy"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (fld, title) in zip(axes, fields):
        im = ax.imshow(to_numpy(fld[:, :, 0]).T, origin="lower", cmap="jet")
        ax.set_title(title)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("metric fields on the reference cube (z-slice)")
    fig.tight_layout()
    return fig


def plot_tissue_map(grid, tissue):
    """The realised tissue labelling against the intended interfaces, and the
    radial conductivity profile.

    Left: the dominant membership at each node on the physical disc, with the
    intended interfaces drawn as circles. The gap between the dashed circles and
    colour boundaries includes node quantisation and mapped-grid spacing: the label
    drawn is whichever tissue has the largest weight at a node. The solve does not
    use this hard label; it uses the smooth weights whose argmax is displayed.

    Right: both conductivities along the +x ray from the axis, on a log scale
    because the default limb values span a large range. For `LIMB`, the two curves
    separate only in muscle, its only anisotropic layer.
    """
    X, Y = grid.coords
    idx = to_numpy(tissue.layer_weights(grid).argmax(0)[:, :, 0])
    sigma_T, sigma_L = tissue.sigma_fields(grid)
    o = grid.centre
    radii = [
        grid.physical_shell_radius(R, tissue.shell_norm)
        for R in tissue.interfaces
    ]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    cmap, bnorm = _tissue_cmap(tissue.layer_names)
    axL.set_facecolor("0.92")
    axL.pcolormesh(
        cell_corners(X),
        cell_corners(Y),
        idx,
        cmap=cmap,
        norm=bnorm,
        shading="flat",
    )
    axL.plot(*_rim_path(X, Y), color="k", lw=0.8)
    for r in radii:
        axL.add_patch(Circle((o, o), r, fill=False, ec="k", lw=1.0, ls="--"))
    axL.set_aspect("equal")
    axL.set_xlabel("x")
    axL.set_ylabel("y")
    axL.set_title(
        f"tissue map, z-slice (n = {grid.n})\n"
        f"dashed = intended interface, colour = realised label"
    )
    axL.legend(
        handles=_tissue_legend_handles(tissue.layer_names),
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    mid = grid.n // 2
    xs = to_numpy(X)[mid:, mid]
    axR.step(
        xs,
        to_numpy(sigma_T)[mid:, mid, 0],
        where="mid",
        color="tab:blue",
        label=r"$\sigma_T$",
    )
    axR.step(
        xs,
        to_numpy(sigma_L)[mid:, mid, 0],
        where="mid",
        color="tab:orange",
        label=r"$\sigma_L$",
    )
    for r in radii:
        axR.axvline(o + r, color="k", lw=0.8, ls="--")
    axR.set_yscale("log")
    axR.set_xlabel("physical $x$ along the $+x$ ray from the axis")
    axR.set_ylabel("conductivity  (S/m)")
    axR.set_title(
        "radial conductivity profile\n" "only muscle separates the two curves"
    )
    axR.legend()

    fig.tight_layout()
    return fig


# ---- the inversion ----------------------------------------------------------


def plot_convergence(
    sigma_history,
    sigma_true,
    param_names,
    data_loss,
    noise_percent,
    reg_loss=None,
    total_loss=None,
    style=None,
):
    """Three figures: the objective, the iterates, and their relative error.

    Returned as a tuple `(fig_loss, fig_sigma, fig_error)` rather than as one
    figure with three panels, because they are read at different times -- the
    first says whether the optimiser is working, the other two say what it
    recovered -- and they carry different aspect ratios.

    `reg_loss` and `total_loss` omitted draws the unregularised case, where the
    data misfit is the whole objective and a second curve would be the same line.

    Both parameter figures use a log ordinate, which accommodates conductivity
    values and relative errors spanning several orders of magnitude.
    """
    style = PARAM_STYLE if style is None else style
    sig = _series(sigma_history)
    true = _series(sigma_true)

    if sig.shape[1] != len(param_names):
        raise ValueError(
            f"sigma_history has {sig.shape[1]} columns but "
            f"{len(param_names)} parameters were named"
        )

    fig_loss, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(
        _series(data_loss),
        color="0.2",
        label="data misfit at the electrode nodes",
    )
    if reg_loss is not None:
        ax.semilogy(
            _series(reg_loss), color="0.2", ls="--", label="Tikhonov penalty"
        )
    if total_loss is not None:
        ax.semilogy(
            _series(total_loss),
            color="C3",
            ls=":",
            label=r"total objective $J$",
        )
    ax.legend()
    ax.set_xlabel("Iterations (L-BFGS)")
    ax.set_ylabel("loss")
    ax.set_title(f"data fit misfit ({noise_percent:.1f}% noise)")
    fig_loss.tight_layout()

    fig_sigma, ax = plt.subplots(figsize=(9, 4.5))
    for k, nm in enumerate(param_names):
        colour, ls = style[nm]
        ax.semilogy(
            sig[:, k], color=colour, ls=ls, label=rf"$\hat\sigma$ {nm}"
        )
        ax.axhline(true[k], color=colour, ls=":", alpha=0.7)
    ax.legend(ncol=2, fontsize=9)
    ax.set_xlabel("Iterations (L-BFGS)")
    ax.set_ylabel("conductivity (S/m)")
    ax.set_title(
        f"conductivity recovery, dotted = truth ({noise_percent:.1f}% noise)"
    )
    fig_sigma.tight_layout()

    fig_error, ax = plt.subplots(figsize=(9, 4.5))
    for k, nm in enumerate(param_names):
        colour, ls = style[nm]
        ax.semilogy(
            np.abs(sig[:, k] - true[k]) / true[k],
            color=colour,
            ls=ls,
            label=nm,
        )
    ax.legend(ncol=2, fontsize=9)
    ax.set_xlabel("Iterations (L-BFGS)")
    ax.set_ylabel("relative parameter error")
    ax.set_title(f"parameter convergence ({noise_percent:.1f}% noise)")
    fig_error.tight_layout()

    return fig_loss, fig_sigma, fig_error


def plot_ensemble_misfit(
    loss_histories, noise_percent, labels=None, floor=None, restart_at=None
):
    """The data misfit of several inversions of the same data, one line per run.

    The runs differ only in where they started, so the figure reads as the spread
    the starting point is responsible for: the curves begin decades apart, at
    whatever misfit each draw happened to have, and the question the figure answers
    is whether they end together. A shared log ordinate is what makes that legible,
    and is why the starting misfits are not normalised away -- how far a run had to
    come is part of the result.

    The abscissa counts objective evaluations rather than L-BFGS iterations. That is
    what the histories record, and a line search makes the two differ; calling it
    iterations would overstate how cheap a run was. It is logarithmic, which gives
    each decade of effort the same width. On a linear axis a run that descends over
    its first fifty evaluations and then sits at the noise floor for six hundred
    spends nine tenths of the figure on the part where nothing moves, and the part
    that carries the result is compressed into the left margin. Nothing is dropped
    or rescaled to achieve that -- the same points are drawn, at the same values --
    but the compression now falls on the plateau rather than on the descent, so the
    `floor` reference below is what keeps the plateau readable.

    Runs are not required to be the same length. A shorter history only means that
    its optimiser made fewer objective evaluations; termination diagnostics must
    be consulted separately.

    Parameters
    ----------
    floor      : the misfit at the true parameters, drawn as a horizontal rule.
                 This is a reference value, not a lower bound: a fitted model can
                 fall below the truth's misfit by fitting part of the realised
                 noise. The line shows where the data-generating parameters lie on
                 the same objective scale.
    restart_at : one sequence of history indices per run, marking where L-BFGS-B
                 was restarted. The misfit is flat across a boundary by
                 construction, because the next call re-evaluates the point it was
                 handed. Marking the duplicated boundary point distinguishes a
                 restart from an ordinary flat segment.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    for k, history in enumerate(loss_histories):
        y = _series(history)
        # evaluations counted from one, not zero: the first evaluation is the
        # first unit of work done, and a log abscissa has no zero to put it at
        x = np.arange(1, len(y) + 1)
        colour = f"C{k % 10}"
        ax.loglog(
            x,
            y,
            color=colour,
            lw=1.2,
            label=f"run {k + 1}" if labels is None else labels[k],
        )
        if restart_at is not None:
            where = np.asarray(restart_at[k], dtype=int)
            if len(where):
                ax.plot(
                    x[where],
                    y[where],
                    linestyle="none",
                    marker="o",
                    ms=3.5,
                    mfc="none",
                    mec=colour,
                    mew=1.0,
                )

    # Drawn behind the curves and keyed to the shared data-generating parameters,
    # not to any individual optimiser run.
    if floor is not None:
        ax.axhline(
            float(floor),
            color="0.35",
            lw=1.0,
            ls="--",
            zorder=0,
            label="misfit at the truth",
        )

    handles, legend_labels = ax.get_legend_handles_labels()
    if restart_at is not None and any(len(w) for w in restart_at):
        # one entry for the marker, in ink rather than in any run's colour: it
        # means the same thing on every curve
        handles.append(
            plt.Line2D(
                [], [], linestyle="none", marker="o", ms=3.5,
                mfc="none", mec="0.35", mew=1.0,
            )
        )
        legend_labels.append("L-BFGS-B restart")
    ax.legend(handles, legend_labels, ncol=2, fontsize=9)

    ax.set_xlabel("objective evaluations (L-BFGS-B)")
    ax.set_ylabel("data misfit $J$")
    ax.set_title(
        f"data misfit, {len(loss_histories)} starting points "
        f"({noise_percent:.1f}% noise)"
    )
    fig.tight_layout()
    return fig


def plot_error_vs_noise(
    noise_percents,
    mean_error,
    floor,
    param_names,
    layer_names,
    spread=None,
):
    r"""Relative recovery error against measurement noise, with the noise-free
    error drawn as a zero-noise reference.

    Log on both axes. The noise levels are a geometric spread rather than an
    arithmetic one -- the interesting comparison is 1% against 0.25%, a factor of
    four, and not the 0.75 percentage points between them -- and a relative error
    is a ratio, so both quantities are read multiplicatively and both axes are
    logarithmic. The slope is then the exponent: a curve at unit slope is an error
    proportional to the noise, and a flat curve is an error the noise is not
    setting.

    The noise-free run is a horizontal rule rather than a point because a
    logarithmic abscissa has no location for zero. It provides a baseline for the
    discretisation, finite-cycle solve, optimisation and parameterisation errors in
    the corresponding zero-noise experiment; it is not asserted to be a strict
    asymptote or lower bound for noisy runs.

    Three panels separate the aggregate, conductivities and thicknesses. All are
    relative errors, but the grouped panels keep the individual series legible and
    make the composition of the aggregate explicit.

    Per-series floors are dotted in the series colour, which is the same idiom
    `plot_convergence` uses for a parameter's true value: a horizontal reference
    keyed to one curve, in that curve's colour, drawn behind it.

    Parameters
    ----------
    noise_percents : the noise levels, all strictly positive. The zero-noise run is
                     `floor` and is not a point on the abscissa.
    mean_error     : (levels, quantities) mean relative error, as a fraction. The
                     columns are the conductivities named by `param_names`
                     followed by the thicknesses named by `layer_names`, which is
                     the order the inversion's parameter vector induces.
    floor          : (quantities,) the same errors at zero noise, same column order.
    spread         : optional (levels,) uncertainty for the all-quantity average,
                     drawn as error bars on the first panel. It is
                     the only place a spread is drawn: the per-parameter panels
                     carry eight curves between them and error bars on all of them
                     would be a thicket. Which spread it is -- the sample standard
                     represented -- for example, a sample standard deviation or a
                     standard error -- is the caller's to state in the caption.
    """
    x = np.asarray([float(p) for p in noise_percents], dtype=float)
    if x.size == 0 or np.any(x <= 0):
        raise ValueError(
            "the abscissa is logarithmic, so every level drawn on it must be "
            f"positive; the zero-noise run belongs in `floor`, got {tuple(x)}"
        )

    # percent, so the axes read as the tables do
    err = 100.0 * _series(mean_error).astype(float)
    flr = 100.0 * _series(floor).astype(float).reshape(-1)
    names = list(param_names) + list(layer_names)

    if err.shape != (x.size, len(names)):
        raise ValueError(
            f"mean_error is {err.shape} against {x.size} noise levels and "
            f"{len(names)} named quantities"
        )
    if flr.size != len(names):
        raise ValueError(
            f"floor has {flr.size} entries against {len(names)} named quantities"
        )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    n_sigma = len(param_names)

    # ---- the aggregate ------------------------------------------------------
    ax = axes[0]
    aggregate = err.mean(1)
    if spread is None:
        ax.plot(x, aggregate, color="0.2", lw=1.8, marker="o", ms=5.5)
    else:
        ax.errorbar(
            x,
            aggregate,
            yerr=100.0 * _series(spread).astype(float).reshape(-1),
            color="0.2",
            lw=1.8,
            marker="o",
            ms=5.5,
            capsize=3,
            elinewidth=1.0,
        )
    ax.axhline(flr.mean(), color="0.35", lw=1.0, ls="--", zorder=0)
    ax.annotate(
        "noise-free floor",
        xy=(x[0], flr.mean()),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color="0.35",
    )
    ax.set_title(f"all {len(names)} quantities")

    # ---- the two groups, each in the tissue palette -------------------------
    groups = (
        (
            axes[1],
            list(param_names),
            range(n_sigma),
            [PARAM_STYLE[nm] for nm in param_names],
            "conductivities",
        ),
        (
            axes[2],
            list(layer_names),
            range(n_sigma, len(names)),
            [(TISSUE_COLOURS[nm], "-") for nm in layer_names],
            "layer thicknesses",
        ),
    )
    for ax, group_names, columns, styles, title in groups:
        for nm, column, (colour, ls) in zip(group_names, columns, styles):
            ax.plot(
                x,
                err[:, column],
                color=colour,
                ls=ls,
                lw=1.8,
                marker="o",
                ms=4.5,
                label=nm,
            )
            # this series' own floor, behind it and in its colour
            ax.axhline(
                flr[column], color=colour, ls=":", lw=1.0, alpha=0.8, zorder=0
            )
        # placed where it collides least: five rising curves leave different
        # corners empty in the two panels, and a fixed corner would sit on one
        # of them. Framed, because a transparent legend over a pale series is
        # unreadable in both directions.
        ax.legend(fontsize=8, loc="best", framealpha=0.85, edgecolor="0.85")
        ax.set_title(title)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("measurement noise (% of clean RMS)")
        # on every panel, not only the leftmost: the three carry different
        # ordinate ranges and are not a shared axis, so a single label on the
        # left would be read as covering scales it does not
        ax.set_ylabel("mean relative error (%)")
        # the levels themselves, labelled. A log locator over a span this short
        # offers one decade tick and a crowd of unlabelled minors, neither of
        # which names a level the experiment actually ran.
        ax.xaxis.set_major_locator(FixedLocator(x))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        # 1, 2, 5 per decade and labelled; the ordinate spans anything from a
        # fraction of a percent to tens of percent across the three panels, and
        # a decade tick alone leaves a panel inside one decade with two labels
        # while labelling every log minor leaves it with nine.
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(True, which="both", lw=0.4, alpha=0.25)
        ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_table(headers, rows, title=None, rule_after=(), fontsize=11):
    """A table of already-formatted strings, drawn as a figure.

    A table is not a chart. Nothing here is worth encoding as a position or a
    colour -- eight numbers read against their references is exactly the case the
    reader wants to see as digits -- so this draws digits, and spends its effort on
    the things that make a table legible instead: a monospaced face, right-aligned
    numeric columns and
    labels left-aligned, one rule under the header, and no cell fills or gridlines
    to compete with the values. The text is ink-coloured throughout; a table that
    colours its numbers is encoding twice and reading once.

    Formatting is the caller's, not this function's. A figure that decided how many
    digits a conductivity gets would be a second opinion on the printed table it is
    supposed to reproduce, and the two would drift apart.

    Parameters
    ----------
    headers    : one string per column. The first column is treated as the labels
                 and is left-aligned; the rest are numbers and are right-aligned.
    rows       : sequence of same-length string tuples.
    title      : a caption, drawn above the header. Newlines are honoured and a
                 caption wider than the columns widens the figure, so a long one
                 is a layout choice rather than an overflow.
    rule_after : row indices after which to draw a light rule, for a table made of
                 blocks that measure different things.
    """
    ncol = len(headers)
    for k, row in enumerate(rows):
        if len(row) != ncol:
            raise ValueError(
                f"row {k} has {len(row)} entries against {ncol} headers"
            )

    widths = [
        max(len(str(headers[j])), *(len(str(r[j])) for r in rows))
        for j in range(ncol)
    ]
    gap = 2  # characters between columns
    total = sum(widths) + gap * (ncol - 1)

    char_w = fontsize / 72.0 * 0.62
    row_h = fontsize / 72.0 * 1.9
    margin = 1  # one character of air either side, so nothing sits on the crop
    caption = [] if title is None else str(title).splitlines()
    content = max([total] + [len(line) for line in caption])
    height = len(rows) + 1.6 + (len(caption) + 0.4 if caption else 0)

    # a monospaced figure sized in characters: the geometry is then the same at
    # every column width, and nothing has to be nudged by hand. The caption is
    # measured too, so a long one widens the figure instead of running off it
    char_w = fontsize / 72.0 * 0.62
    row_h = fontsize / 72.0 * 1.9
    fig = plt.figure(figsize=((content + 2 * margin) * char_w, height * row_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(-margin, content + margin)
    ax.set_ylim(height, 0)

    edges, x = [], 0
    for w in widths:
        edges.append((x, x + w))
        x += w + gap

    def line(y, values, weight="normal", colour="0.15"):
        for j, (value, (x0, x1)) in enumerate(zip(values, edges)):
            ax.text(
                x0 if j == 0 else x1,
                y,
                str(value),
                ha="left" if j == 0 else "right",
                va="center",
                family="monospace",
                fontsize=fontsize,
                fontweight=weight,
                color=colour,
            )

    y = 0.6
    for line_text in caption:
        ax.text(
            0,
            y,
            line_text,
            ha="left",
            va="center",
            family="monospace",
            fontsize=fontsize,
            color="0.35",
        )
        y += 1
    if caption:
        y += 0.4

    line(y, headers, weight="bold")
    ax.plot([0, total], [y + 0.5] * 2, color="0.25", lw=1.0)

    for k, row in enumerate(rows):
        line(y + 1 + k, row)
        if k in rule_after and k != len(rows) - 1:
            ax.plot([0, total], [y + 1.5 + k] * 2, color="0.85", lw=0.8)
    return fig


def _thicknesses(interfaces):
    """Interface radii -> layer thicknesses, along the last axis.

    Thicknesses give one value per tissue and are often easier to interpret than
    interface coordinates. They also let these figures share the tissue palette
    used by the map and animation.
    """
    R = np.atleast_2d(_series(interfaces))
    inner = np.concatenate([np.zeros_like(R[:, :1]), R], axis=1)
    return np.diff(
        np.concatenate([inner, np.full_like(R[:, :1], 0.5)], axis=1), axis=1
    )


def plot_radius_convergence(
    interface_history,
    interfaces_true,
    layer_names,
    noise_percent,
    limb_radius=None,
    unit="mm",
):
    """Two figures: the recovered layer thicknesses, and their relative error.

    The radius analogue of the two parameter figures `plot_convergence` returns, and
    deliberately separate from them. Thicknesses are O(0.1) in normalised units where
    the conductivities are O(0.01) S/m, and the two carry different dimensions, so a
    shared ordinate would be a category error however convenient the axis looked.

    `limb_radius` is the physical radius of the limb, given in whatever `unit` the
    axis should read in -- 24 and "mm" for a 2.4 cm forearm. Omitted, the figures are
    in normalised units, where the skin surface sits at 0.5. The scale is an argument
    because the model has none: the transverse geometry is normalised, so millimetres
    enter only when a caller says what the limb measures.

    A layer held fixed rather than recovered has zero error at every iterate and no
    logarithm, so it is drawn in the thickness figure -- where holding it is worth
    seeing -- and dropped from the error figure.
    """
    thick = _thicknesses(interface_history)
    true = _thicknesses(interfaces_true)[0]

    if thick.shape[1] != len(layer_names):
        raise ValueError(
            f"the history gives {thick.shape[1]} thicknesses but "
            f"{len(layer_names)} layers were named"
        )

    scale = 1.0 if limb_radius is None else limb_radius / 0.5
    label = "normalised" if limb_radius is None else unit
    thick, true = thick * scale, true * scale

    fig_thick, ax = plt.subplots(figsize=(9, 4.5))
    for k, nm in enumerate(layer_names):
        ax.semilogy(
            thick[:, k], color=TISSUE_COLOURS[nm], label=rf"$\hat t$ {nm}"
        )
        ax.axhline(true[k], color=TISSUE_COLOURS[nm], ls=":", alpha=0.7)
    ax.legend(ncol=2, fontsize=9)
    ax.set_xlabel("Iterations (L-BFGS)")
    ax.set_ylabel(f"layer thickness ({label})")
    ax.set_title(
        f"thickness recovery, dotted = truth ({noise_percent:.1f}% noise)"
    )
    fig_thick.tight_layout()

    fig_error, ax = plt.subplots(figsize=(9, 4.5))
    for k, nm in enumerate(layer_names):
        err = np.abs(thick[:, k] - true[k]) / true[k]
        if not np.any(err > 0):
            continue  # held fixed: nothing to show on a log axis
        ax.semilogy(err, color=TISSUE_COLOURS[nm], label=nm)
    ax.legend(ncol=2, fontsize=9)
    ax.set_xlabel("Iterations (L-BFGS)")
    ax.set_ylabel("relative thickness error")
    ax.set_title(f"thickness convergence ({noise_percent:.1f}% noise)")
    fig_error.tight_layout()

    return fig_thick, fig_error


def _panel_grid(n_panels):
    """(fig, flat axes) for `n_panels` square panels, at most four to a row."""
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False
    )
    return fig, axes, axes.ravel()


def plot_field_panels(observation, snapshots, z_fraction, noise_percent=None):
    """The observed field and each stored prediction, on one cross-section.

    `observation` supplies the data and the grid it was measured on; `snapshots` is
    a sequence of `(iteration, phi, relative misfit)` triples, and this function
    does not care how they were produced.

    Every panel shares one symmetric colour scale, taken from the observed field
    over the whole volume rather than per panel, so that the panels are comparable
    to each other: a per-panel scale would make an early iterate with a tenth of
    the amplitude look like a good fit.
    """
    grid = observation.grid
    k = slice_index(grid.n, z_fraction)
    X, Y = grid.coords
    phi_obs = to_numpy(observation.phi)
    vmax = np.abs(phi_obs).max()

    fig, axes, flat = _panel_grid(1 + len(snapshots))

    im = plot_disc(
        flat[0],
        phi_obs[:, :, k],
        X,
        Y,
        vmax,
        f"observed $\\phi$ (data, n = {grid.n})",
    )
    for ax, (it, phi_pred, rel_misfit) in zip(flat[1:], snapshots):
        im = plot_disc(
            ax,
            to_numpy(phi_pred)[:, :, k],
            X,
            Y,
            vmax,
            f"$\\phi$ predicted @ iter {it}\nrel misfit = {rel_misfit:.3f}",
        )
    for ax in flat[1 + len(snapshots) :]:
        ax.axis("off")

    noise = (
        ""
        if noise_percent is None
        else f", {noise_percent:.1f}% measurement noise"
    )
    fig.suptitle(
        f"data-fit evolution -- z = {z_fraction:.3f} L (index {k}), "
        f"{grid.n}$^3$ grid{noise}"
    )
    fig.colorbar(im, ax=axes, label=r"$\phi$", shrink=0.8)
    return fig


def plot_residual_panels(
    observation, snapshots, z_fraction, noise_percent=None
):
    """The residual (predicted minus observed) for each stored prediction.

    The colour scale is symmetric and shared, taken over every residual in the
    whole volume, so the panels show the residual shrinking rather than each being
    renormalised to look the same. The displayed residual can contain measurement
    noise, modelling error, solver error and imperfect optimisation; the panels
    show its spatial structure without assigning those causes automatically.
    """
    if len(snapshots) == 0:
        raise ValueError(
            "no snapshots to plot: there is no residual without a prediction"
        )

    grid = observation.grid
    k = slice_index(grid.n, z_fraction)
    X, Y = grid.coords
    phi_obs = to_numpy(observation.phi)

    residuals = [
        (it, to_numpy(phi_pred) - phi_obs, rel_misfit)
        for (it, phi_pred, rel_misfit) in snapshots
    ]
    vmax = max(np.abs(d).max() for (_, d, _) in residuals)

    fig, axes, flat = _panel_grid(len(residuals))

    for ax, (it, d, rel_misfit) in zip(flat, residuals):
        im = plot_disc(
            ax,
            d[:, :, k],
            X,
            Y,
            vmax,
            f"$\\phi$ predicted $-$ observed @ iter {it}\n"
            f"rel misfit = {rel_misfit:.3f}",
        )
    for ax in flat[len(residuals) :]:
        ax.axis("off")

    noise = (
        ""
        if noise_percent is None
        else f", {noise_percent:.1f}% measurement noise"
    )
    fig.suptitle(
        f"residual (predicted $-$ observed) -- z = {z_fraction:.3f} L "
        f"(index {k}), {grid.n}$^3$ grid{noise}"
    )
    fig.colorbar(
        im, ax=axes, label=r"$\phi$ predicted $-$ observed", shrink=0.8
    )
    return fig


MAX_GEOMETRY_PANELS = 24


def plot_geometry_panels(
    grid, tissue, snapshots, epsilon, z_fraction=0.5, snapshot_every=None
):
    """The recovered limb cross-section at each stored snapshot, against the truth.

    The spatial counterpart of `plot_radius_convergence`: those figures say how the
    thicknesses converged, this one says what the limb looked like while they did.

    `snapshots` is a sequence of `(evaluation, sigma_T vector, interface radii)`
    triples. The field is rebuilt here from the per-layer transverse conductivities
    and interface radii rather than stored per panel, avoiding a full 3-D field for
    every snapshot. Because a snapshot is
    that cheap, the intended use is to record every evaluation and choose the spacing
    here -- `snapshot_every` draws every k-th stored entry, so the figure can be redrawn at a
    different spacing without rerunning the inversion.

    The last snapshot is always drawn whether or not the spacing lands on it, so
    thinning cannot omit the returned end of the recorded history.

    The number of selected snapshots is capped to avoid producing an unreadably
    large figure; the truth panel is added separately. Use `snapshot_every` to thin
    a longer history.

    Solid circles are the recovered interfaces, dashed are the true ones, so a panel
    shows the layers moving and the gap closing. The colour is sigma_T on a shared
    logarithmic scale over every panel including the truth -- shared because a
    per-panel scale would renormalise an early iterate with the wrong contrast into
    looking correct. As in the radial profile, the logarithmic scale is useful
    because the tissue table spans a factor of fifty from bone to skin; a linear
    ramp would compress the lower-conductivity layers into a narrow part of the
    colour range.
    """
    if len(snapshots) == 0:
        raise ValueError(
            "no snapshots to plot: there is no geometry without an iterate"
        )

    recorded = len(snapshots)
    if snapshot_every is not None:
        if snapshot_every < 1:
            raise ValueError(
                f"snapshot_every must be at least 1, got {snapshot_every}"
            )
        drawn = list(snapshots[::snapshot_every])
        if drawn[-1] is not snapshots[-1]:
            drawn.append(snapshots[-1])
        snapshots = drawn
    if len(snapshots) > MAX_GEOMETRY_PANELS:
        raise ValueError(
            f"{len(snapshots)} snapshots would draw {1 + len(snapshots)} panels, over the "
            f"{MAX_GEOMETRY_PANELS} this figure is readable at. Pass snapshot_every to "
            f"thin them -- snapshot_every={-(-recorded // MAX_GEOMETRY_PANELS) + 1} "
            f"would draw about {MAX_GEOMETRY_PANELS} of the {recorded} recorded."
        )

    X, Y = grid.coords
    k = slice_index(grid.n, z_fraction)
    o = grid.centre
    true_radii = [
        grid.physical_shell_radius(R, tissue.shell_norm)
        for R in tissue.interfaces
    ]

    sT_true, _ = tissue.sigma_fields(grid)
    fields = [(None, to_numpy(sT_true)[:, :, k], tissue.interfaces)]
    for it, sT_vec, interfaces in snapshots:
        v = torch.as_tensor(_series(sT_vec))
        R = torch.as_tensor(_series(interfaces))
        with torch.no_grad():
            sT, _ = tissue.sigma_fields(
                grid, sT_vec=v, sL_vec=v, interfaces=R, epsilon=epsilon
            )
        fields.append((it, to_numpy(sT)[:, :, k], _series(interfaces)))

    vmin = min(f.min() for _, f, _ in fields)
    vmax = max(f.max() for _, f, _ in fields)

    # Shared across panels, and logarithmic. Skin at 1.00 S/m against bone at 0.02
    # is a factor of fifty, so a linear ramp puts every tissue but skin inside its
    # bottom tenth. The colour is here to show which layer is which and how the
    # interfaces move, and log is what makes that visible across the range the
    # tissue table actually spans. Positive by construction -- sigma is an
    # exponential of the parameters and a convex combination of positive layer
    # values -- so LogNorm's vmin > 0 holds.
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes, flat = _panel_grid(len(fields))
    for ax, (it, field, radii) in zip(flat, fields):
        ax.set_facecolor("0.92")
        im = ax.pcolormesh(
            cell_corners(X),
            cell_corners(Y),
            field,
            cmap="magma",
            norm=norm,
            shading="flat",
        )
        ax.plot(*_rim_path(X, Y), color="k", lw=0.8)
        for R in true_radii:
            ax.add_patch(
                Circle((o, o), R, fill=False, ec="k", lw=1.0, ls="--")
            )
        if it is not None:
            for R in radii:
                ax.add_patch(
                    Circle(
                        (o, o),
                        grid.physical_shell_radius(R, tissue.shell_norm),
                        fill=False,
                        ec="w",
                        lw=1.2,
                    )
                )
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("truth" if it is None else f"recovered @ iter {it}")
    for ax in flat[len(fields) :]:
        ax.axis("off")

    sampled = (
        ""
        if len(snapshots) == recorded
        else f", {len(snapshots)} of {recorded} iterates"
    )
    fig.suptitle(
        f"limb cross-section -- z = {z_fraction:.3f} L (index {k}), "
        f"{grid.n}$^3$ grid{sampled}\n"
        f"white = recovered interface, dashed = truth"
    )
    fig.colorbar(im, ax=axes, label=r"$\sigma_T$  (S/m)", shrink=0.8)
    return fig


def animate_slices(
    sources,
    fields,
    grid,
    tissue,
    noise_percent=None,
    interval=120,
    labels=None,
):
    """Sweep the cross-section along the limb axis: every source over the tissue
    shells on the left, one measured-potential panel per source on the right.

    `sources` and `fields` are matching sequences, one entry per source; a bare
    tensor is accepted for either and treated as a single-source experiment. The
    potentials are not summed: each supplied field is drawn as the separate solve
    paired with its source.

    Returns `(figure, FuncAnimation)`. Rendering is left to the caller -- a
    notebook turns the animation into HTML, a script writes a file -- so that
    IPython is not a dependency of this package.

    The source is drawn as contours over a filled tissue map rather than as a
    second filled field: filling either one hides the other, and the point of the
    panel is that the blob sits inside muscle and clears both of its interfaces.
    Positive and negative lobes are distinguished by colour and linestyle, so the
    dipole reads as a dipole even in a single frame.

    Every colour scale is taken over the whole volume of its own field, so a slice
    away from the blob correctly looks empty instead of being renormalised into
    looking like a source, and a weaker source is not flattened by a stronger one.
    """
    X, Y = grid.coords
    I_list = [to_numpy(a) for a in _as_sequence(sources)]
    phi_list = [to_numpy(a) for a in _as_sequence(fields)]
    if len(I_list) != len(phi_list):
        raise ValueError(
            f"{len(I_list)} sources against {len(phi_list)} fields: the "
            f"animation draws one potential panel per source, so an "
            f"unpaired field is a source whose solve is missing"
        )
    n_src = len(I_list)
    labels = (
        [f"S{s + 1}" for s in range(n_src)] if labels is None else list(labels)
    )

    n = grid.n
    vmax_I = [np.abs(a).max() for a in I_list]
    vmax_phi = [np.abs(a).max() for a in phi_list]
    o = grid.centre
    idx = to_numpy(tissue.layer_weights(grid).argmax(0)[:, :, 0])
    radii = [
        grid.physical_shell_radius(R, tissue.shell_norm)
        for R in tissue.interfaces
    ]

    fig, axes = plt.subplots(
        1, 1 + n_src, figsize=(5.5 * (1 + n_src), 4.8), squeeze=False
    )
    axI, axes_phi = axes[0, 0], axes[0, 1:]

    cmap, bnorm = _tissue_cmap(tissue.layer_names)
    axI.set_facecolor("0.85")
    axI.pcolormesh(
        cell_corners(X),
        cell_corners(Y),
        idx,
        cmap=cmap,
        norm=bnorm,
        shading="flat",
    )
    for r in radii:
        axI.add_patch(Circle((o, o), r, fill=False, ec="k", lw=1.0))
    axI.plot(
        *_rim_path(X, Y), color="k", lw=1.0
    )  # the rim IS the skin-air interface
    axI.set_aspect("equal")
    axI.set_xlabel("x")
    axI.set_ylabel("y")
    axI.legend(
        handles=_tissue_legend_handles(tissue.layer_names),
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    meshes = []
    for s, ax in enumerate(axes_phi):
        pm = plot_disc(ax, phi_list[s][:, :, 0], X, Y, vmax_phi[s], "")
        for r in radii:
            ax.add_patch(
                Circle((o, o), r, fill=False, ec="k", lw=0.9, alpha=0.65)
            )
        fig.colorbar(pm, ax=ax, label=r"$\phi$")
        meshes.append(pm)

    # Static tag at each blob's transverse position, read off the source field itself
    # rather than from a placement argument, so it cannot disagree with what is drawn.
    for s, I_np in enumerate(I_list):
        w = np.abs(I_np).sum(axis=2)
        i, j = np.unravel_index(int(w.argmax()), w.shape)
        axI.annotate(
            labels[s],
            (X[i, j], Y[i, j]),
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="0.4", alpha=0.85),
        )

    noise = "" if noise_percent is None else f" ({noise_percent:.1f}% noise)"
    levels = [v * np.array([0.15, 0.5, 0.85]) for v in vmax_I]
    drawn = []  # this frame's contour sets, cleared in place by the next frame

    def update(k):
        for cs in drawn:
            cs.remove()
        drawn.clear()

        for s, I_np in enumerate(I_list):
            Ik, lv = I_np[:, :, k], levels[s]
            if Ik.max() > lv[0]:
                drawn.append(
                    axI.contour(
                        X, Y, Ik, levels=lv, colors="darkred", linewidths=1.2
                    )
                )
            if Ik.min() < -lv[0]:
                drawn.append(
                    axI.contour(
                        X,
                        Y,
                        Ik,
                        levels=-lv[::-1],
                        colors="darkblue",
                        linewidths=1.2,
                        linestyles="dashed",
                    )
                )
            meshes[s].set_array(phi_list[s][:, :, k].ravel())

        z = k * grid.L / (n - 1)
        axI.set_title(f"tissue shells + sources   z = {z:.3f}  (k = {k})")
        for s, ax in enumerate(axes_phi):
            ax.set_title(f"observed $\\phi$, {labels[s]}{noise}   z = {z:.3f}")
        return tuple(meshes)

    anim = animation.FuncAnimation(
        fig, update, frames=n, interval=interval, blit=False
    )
    return fig, anim
