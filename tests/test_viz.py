# I used Claude Code to help implement and package tests for viz.py.

"""Tests for neuralphysics.viz.

Plotting code admits few assertions that can fail for an interesting reason, so
these check the things that would silently produce a wrong figure: the geometry
helpers, which plane a depth fraction selects, how many panels are drawn, and
that project-specific runtime data are not read from module scope. Matplotlib's
own style state and pixel output are not tested.
"""

import inspect
import types

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use(
    "Agg"
)  # no display in CI; must precede the pyplot import in viz

from matplotlib.figure import Figure

from neuralphysics import viz
from neuralphysics.data import Observation, lateral_surface_mask
from neuralphysics.grid import grid_at
from neuralphysics.tissue import LIMB

# Fixed plotting inputs for the bundled limb: five conductivity labels and a
# starting pair of free inner radii. The tests need names and a geometry distinct
# from the truth, not values derived through the parameterisation helpers.
PARAM_NAMES = ["bone", "muscle_T", "muscle_L", "fat", "skin"]
RADII_START = (8.0 / 48.0, 19.5 / 48.0)


@pytest.fixture(scope="module")
def small():
    """A 9-node grid and the limb model: the coarsest level at which every shell is
    non-empty, so the tissue figures are meaningful."""
    return grid_at(9), LIMB


def _z_indexed_observation(n):
    """An observation whose value at every node equals that node's z index, so a
    figure drawn from it identifies the plane it was drawn on."""
    grid = grid_at(n)
    phi = torch.arange(n, dtype=torch.float64).expand(n, n, n).clone()
    return Observation(phi=phi, mask=lateral_surface_mask(n), grid=grid)


def _panels(fig):
    """The drawing axes of a panel figure, with the colorbar axis excluded."""
    return [
        ax
        for ax in fig.axes
        if ax.get_label() != "<colorbar>" and ax.collections
    ]


def _mesh_values(ax):
    """The scalar field a pcolormesh panel is carrying, as a flat array."""
    return np.asarray(ax.collections[0].get_array()).ravel()


def test_cell_corners_are_the_midpoints_of_their_four_nodes(configured):
    """Interior corners must average the four surrounding nodes. Checked against a
    linear field, where the average has a closed form and any indexing slip moves
    the answer by a whole cell."""
    n = 6
    A = np.arange(n)[:, None] * np.ones((1, n)) * 10.0 + np.arange(n)[None, :]
    C = viz.cell_corners(A)
    assert C.shape == (n + 1, n + 1)
    for i in range(1, n):
        for j in range(1, n):
            assert C[i, j] == pytest.approx(
                0.25 * (A[i - 1, j - 1] + A[i, j - 1] + A[i - 1, j] + A[i, j])
            )


@pytest.mark.parametrize(
    "n, fraction, expected",
    [
        (65, 0.703125, 45),
        (33, 0.5, 16),
    ],
)
def test_slice_fraction_selects_the_same_physical_plane_at_every_n(
    n, fraction, expected
):
    """The point of the fraction: 0.5 is the mid-plane at every resolution, where a
    fixed index is a different plane at each. 0.703125 is the fraction the
    notebook's hard-coded index 45 corresponded to at N = 65."""
    assert viz.slice_index(n, fraction) == expected


def test_plot_disc_draws_on_the_given_axis_with_a_symmetric_scale(
    configured, small
):
    """The scale must be symmetric about zero: the potential is signed and a
    colormap centred anywhere else misreports the sign of half the field."""
    grid, _ = small
    fig = Figure()
    ax = fig.subplots()
    field = torch.randn(grid.n, grid.n, dtype=torch.float64)

    pm = viz.plot_disc(ax, field, *grid.coords, 0.7, "title")

    assert pm.get_clim() == (-0.7, 0.7)
    assert list(ax.collections) == [pm]
    assert len(ax.lines) == 1  # the rim outline
    assert ax.get_title() == "title"


def test_convergence_accepts_histories_as_plain_lists(configured):
    """Histories accumulate as lists inside the objective, so requiring tensors would
    push a conversion onto every call site."""
    names = PARAM_NAMES
    fig_loss, fig_sigma, _ = viz.plot_convergence(
        [[0.1] * 5, [0.2] * 5],
        [0.02, 0.09, 0.40, 0.04, 0.05],
        names,
        data_loss=[1e-2, 1e-3],
        noise_percent=5.0,
        reg_loss=[1e-7, 2e-7],
    )
    assert len(fig_loss.axes[0].lines) == 2
    assert len(fig_sigma.axes[0].lines) == 2 * len(names)


def test_radius_figures_hold_the_fixed_layer_out_of_the_error_plot(configured):
    """Four thicknesses recovered against four truths, but skin is held fixed and has
    no relative error to put on a log axis, so it appears in one figure and not the
    other. Drawing it in both would print a flat line at zero and silently rescale
    the axis the recovered layers are read on."""
    tissue = LIMB
    history = np.linspace(
        np.array(RADII_START + (tissue.interfaces[-1],)),
        np.array(tissue.interfaces),
        12,
    )

    fig_thick, fig_error = viz.plot_radius_convergence(
        history,
        tissue.interfaces,
        tissue.layer_names,
        noise_percent=5.0,
        limb_radius=24.0,
    )

    assert all(isinstance(f, Figure) for f in (fig_thick, fig_error))
    assert len(fig_thick.axes[0].lines) == 2 * len(
        tissue.layer_names
    )  # iterates and truths
    assert (
        len(fig_error.axes[0].lines) == len(tissue.layer_names) - 1
    )  # skin held
    # the last iterate is the truth, in millimetres on a 24 mm limb
    assert viz._thicknesses(history)[-1] * 48.0 == pytest.approx(
        [7.0, 13.0, 3.0, 1.0]
    )


def test_geometry_panels_thin_the_history_but_keep_the_last_iterate(
    configured, small
):
    """Recording every iterate and thinning at plot time is the intended workflow, so
    the spacing has to be an argument here rather than a decision made during the run.
    The final iterate is drawn whether or not the spacing lands on it, because it is
    the result; over the panel cap the call refuses rather than building a figure tall
    enough to exhaust memory."""
    grid, tissue = small
    sT_vec, _ = tissue.sigma_vectors()
    history = [(i, sT_vec, tissue.interfaces) for i in range(31)]

    fig = viz.plot_geometry_panels(
        grid, tissue, history, tissue.epsilon, snapshot_every=10
    )
    titles = [ax.get_title() for ax in _panels(fig)]
    assert titles == [
        "truth",
        "recovered @ iter 0",
        "recovered @ iter 10",
        "recovered @ iter 20",
        "recovered @ iter 30",
    ]

    with pytest.raises(ValueError, match="snapshot_every"):
        viz.plot_geometry_panels(grid, tissue, history, tissue.epsilon)


def test_field_panels_draw_the_data_and_every_snapshot_on_the_requested_plane(
    configured,
):
    """The first panel is the observation and the rest are the predictions, all at
    the plane the fraction resolves to. The field carries its own z index, so the
    panel values prove which plane was drawn -- the failure the hard-coded index
    used to cause silently."""
    n = 9
    obs = _z_indexed_observation(n)
    snapshots = [(20, obs.phi + 1.0, 0.3), (40, obs.phi + 0.5, 0.1)]

    fig = viz.plot_field_panels(
        obs, snapshots, z_fraction=0.5, noise_percent=5.0
    )
    panels = _panels(fig)

    assert len(panels) == 1 + len(snapshots)
    assert _mesh_values(panels[0]) == pytest.approx((n - 1) // 2)
    assert _mesh_values(panels[1]) == pytest.approx((n - 1) // 2 + 1.0)


def test_residual_panels_draw_the_difference_and_share_one_scale(configured):
    """One panel per snapshot, each showing prediction minus observation. A shared
    symmetric scale is what makes the residual visibly shrink between panels; a
    per-panel scale would renormalise every one of them to look identical."""
    n = 9
    obs = _z_indexed_observation(n)
    snapshots = [(20, obs.phi + 2.0, 0.3), (40, obs.phi + 0.5, 0.1)]

    fig = viz.plot_residual_panels(obs, snapshots, z_fraction=0.25)
    panels = _panels(fig)

    assert len(panels) == len(snapshots)
    assert _mesh_values(panels[0]) == pytest.approx(2.0)
    assert _mesh_values(panels[1]) == pytest.approx(0.5)
    assert (
        panels[0].collections[0].get_clim()
        == panels[1].collections[0].get_clim()
    )


def test_animation_renders_every_slice_and_reads_z_from_the_grid(
    configured, small
):
    """Serialising to JavaScript invokes every frame, so contour-update failures are
    exposed. The last title also verifies that the updater derives axial position
    from the supplied grid rather than unrelated module state."""
    grid, tissue = small
    n = grid.n
    I = torch.zeros(n, n, n, dtype=torch.float64)
    I[4, 4, 2], I[4, 4, 6] = 1.0, -1.0
    phi = torch.randn(n, n, n, dtype=torch.float64)

    fig, anim = viz.animate_slices(I, phi, grid, tissue, noise_percent=5.0)
    html = anim.to_jshtml()

    assert len(html) > 0
    assert len([ax for ax in fig.axes if ax.get_label() != "<colorbar>"]) == 2
    assert f"z = {grid.L:.3f}" in fig.axes[0].get_title()


def test_no_module_scope_configuration(configured):
    """Only documented palettes, the panel cap and exports may be public
    non-callable module state; no project data or run configuration may leak in."""
    allowed = {
        "TISSUE_COLOURS",
        "PARAM_STYLE",
        "MAX_GEOMETRY_PANELS",
        "__all__",
    }
    leaked = {
        name
        for name, value in vars(viz).items()
        if not name.startswith("_")
        and not callable(value)
        and not isinstance(value, types.ModuleType)
        and name not in allowed
    }
    assert leaked == set()
    assert not any(
        "plt.show" in inspect.getsource(fn)
        for fn in vars(viz).values()
        if inspect.isfunction(fn)
    )


def test_animation_gives_each_source_its_own_potential_panel(
    configured, small
):
    """One panel per source, each showing its own field. Summing the potentials, or
    reusing one mesh for both, would render an identical pair of panels here and
    pass every other assertion in this module."""
    grid, tissue = small
    n = grid.n
    I1 = torch.zeros(n, n, n, dtype=torch.float64)
    I1[4, 4, 2], I1[4, 4, 6] = 1.0, -1.0
    I2 = torch.zeros(n, n, n, dtype=torch.float64)
    I2[2, 2, 2], I2[2, 2, 6] = 1.0, -1.0
    phi1 = torch.full((n, n, n), 2.0, dtype=torch.float64)
    phi2 = torch.full((n, n, n), -3.0, dtype=torch.float64)

    fig, anim = viz.animate_slices([I1, I2], [phi1, phi2], grid, tissue)
    anim.to_jshtml()

    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(panels) == 3
    assert panels[1].collections[0].get_clim() == (-2.0, 2.0)
    assert panels[2].collections[0].get_clim() == (-3.0, 3.0)


def test_the_ensemble_misfit_figure_draws_one_line_per_run():
    """Optimiser histories may differ in length, so the figure must not require
    them to match."""
    histories = [np.array([1.0, 0.4, 0.1]), np.array([2.0, 0.3])]
    fig = viz.plot_ensemble_misfit(histories, noise_percent=5.0)
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    assert len(ax.lines) == len(histories)
    assert ax.get_yscale() == "log"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "run 1",
        "run 2",
    ]


def test_the_ensemble_misfit_figure_takes_the_labels_it_is_given():
    fig = viz.plot_ensemble_misfit(
        [[1.0, 0.5]], noise_percent=1.0, labels=["seed 4"]
    )
    (ax,) = fig.axes
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["seed 4"]


def test_the_table_draws_every_cell_it_is_given():
    headers = ("quantity", "recovered", "true")
    rows = [("bone (S/m)", "0.0184", "0.0200"), ("fat (mm)", "2.95", "3.00")]
    fig = viz.plot_table(headers, rows, title="two lines\nof caption")
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    drawn = [t.get_text() for t in ax.texts]
    assert drawn[:2] == ["two lines", "of caption"]
    assert set(drawn[2:]) == set(headers) | {c for row in rows for c in row}
    assert not ax.axison


def test_a_caption_wider_than_the_columns_widens_the_figure():
    """Otherwise it runs off the crop, which is how a saved figure loses the
    sentence that says what it is."""
    rows = [("a", "1")]
    narrow = viz.plot_table(("x", "y"), rows, title="short")
    wide = viz.plot_table(("x", "y"), rows, title="a" * 80)
    assert wide.get_size_inches()[0] > 4 * narrow.get_size_inches()[0]


def test_a_ragged_row_is_refused():
    """A row with a missing cell would otherwise be drawn silently, one column
    short, and read as a value under the wrong heading."""
    with pytest.raises(ValueError, match="row 1"):
        viz.plot_table(("a", "b"), [("1", "2"), ("3",)])


def test_the_ensemble_misfit_figure_draws_one_line_per_run():
    """Optimiser histories may differ in length, so the figure must not require
    them to match."""
    histories = [np.array([1.0, 0.4, 0.1]), np.array([2.0, 0.3])]
    fig = viz.plot_ensemble_misfit(histories, noise_percent=5.0)
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    assert len(ax.lines) == len(histories)
    assert ax.get_yscale() == "log"


def test_the_table_draws_every_cell_it_is_given():
    headers = ("quantity", "recovered", "true")
    rows = [("bone (S/m)", "0.0184", "0.0200"), ("fat (mm)", "2.95", "3.00")]
    fig = viz.plot_table(headers, rows, title="two lines\nof caption")
    (ax,) = fig.axes
    drawn = [t.get_text() for t in ax.texts]
    assert drawn[:2] == ["two lines", "of caption"]
    assert set(drawn[2:]) == set(headers) | {c for row in rows for c in row}


def test_a_ragged_row_is_refused():
    """A row with a missing cell would otherwise be drawn silently, one column
    short, and read as a value under the wrong heading."""
    with pytest.raises(ValueError, match="row 1"):
        viz.plot_table(("a", "b"), [("1", "2"), ("3",)])


# ---- error against measurement noise ----------------------------------------

NOISE_LEVELS = (0.25, 0.5, 1.0)
THICKNESS_NAMES = ["bone", "muscle", "fat"]
NOISE_FLOOR = np.array([0.40, 0.01, 0.03, 0.04, 0.006, 0.09, 0.07, 0.09])
NOISE_ERROR = np.stack([NOISE_FLOOR * (1 + p) for p in NOISE_LEVELS])


def _noise_panels(fig):
    """The three drawing axes, left to right."""
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def test_error_against_noise_draws_one_curve_and_one_floor_per_quantity():
    """The floor is the point of the figure and is drawn as a separate line from
    the curve it belongs to, so a panel that lost one would still look complete."""
    fig = viz.plot_error_vs_noise(
        NOISE_LEVELS, NOISE_ERROR, NOISE_FLOOR, PARAM_NAMES, THICKNESS_NAMES
    )
    aggregate, sigma, thickness = _noise_panels(fig)

    assert len(aggregate.lines) == 2                      # the average and its floor
    assert len(sigma.lines) == 2 * len(PARAM_NAMES)
    assert len(thickness.lines) == 2 * len(THICKNESS_NAMES)
    assert [t.get_text() for t in sigma.get_legend().get_texts()] == PARAM_NAMES
    for ax in _noise_panels(fig):
        assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")


def test_error_against_noise_reads_in_percent():
    """The tables quote percentages, and a figure beside them quoting fractions
    would be read as the same number a hundred times smaller."""
    fig = viz.plot_error_vs_noise(
        NOISE_LEVELS, NOISE_ERROR, NOISE_FLOOR, PARAM_NAMES, THICKNESS_NAMES
    )
    _, sigma, _ = _noise_panels(fig)
    drawn = sigma.lines[0].get_ydata()          # bone, the first conductivity
    assert np.allclose(drawn, 100.0 * NOISE_ERROR[:, 0])


def test_a_zero_noise_level_on_the_abscissa_is_refused():
    """A zero-noise result cannot be placed on a logarithmic abscissa and must be
    supplied through the separate reference-line argument."""
    with pytest.raises(ValueError, match="positive"):
        viz.plot_error_vs_noise(
            (0.0,) + NOISE_LEVELS,
            np.vstack([NOISE_FLOOR, NOISE_ERROR]),
            NOISE_FLOOR,
            PARAM_NAMES,
            THICKNESS_NAMES,
        )


def test_an_error_array_that_does_not_match_the_names_is_refused():
    """Otherwise a column is drawn under the wrong parameter's colour and label."""
    with pytest.raises(ValueError, match="named quantities"):
        viz.plot_error_vs_noise(
            NOISE_LEVELS,
            NOISE_ERROR[:, :-1],
            NOISE_FLOOR[:-1],
            PARAM_NAMES,
            THICKNESS_NAMES,
        )
