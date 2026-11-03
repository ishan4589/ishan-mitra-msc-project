# I used Claude Code to help implement and package the inverse solver.

r"""The inverse problem itself: one parameter vector, one run, and an ensemble of runs.

Everything above this module is the forward model; this is what sits on top of it.
For the bundled four-layer limb, the recovered values form one unconstrained
vector of five log-conductivities and two softmax logits. The parameter counts are
derived from the supplied tissue model; the transformation enforces positive
conductivities and outward-ordered radii without box bounds.

A single inversion is a function of where it starts. Reporting one run leaves that
dependence unmeasured, so `run_ensemble` repeats the inversion from several draws
of the starting point and averages what comes back. The draws are made once, up
front, by `sample_initial_parameters`, so which starting points the experiment used
does not depend on how many runs finished.

Only summaries are kept. A run's own recovered values are a step towards the average
and not a result of this experiment, so `EnsembleResult` carries the mean recovered
value, its spread across the runs and the mean relative error, and nothing per-run
except what the figures draw -- the loss and geometry histories -- and the optimiser's
exit reports, which let the caller assess the runs included in the average.

L-BFGS-B may be restarted after a call terminates: see `invert`. A run ends when a
restart accepts no step or fails to improve the objective, unless its shared work
budget is exhausted first.

Nothing here prints or plots: the run driver returns numbers and `viz` draws them,
so the same call works from a notebook, a script and a test.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from .forward import forward
from .hierarchy import CoefficientHierarchy
from .tissue import interfaces_from_logits, logits_from_interfaces
from .utils import to_numpy

__all__ = [
    "parameter_names",
    "sigma_indices",
    "true_sigma",
    "true_parameters",
    "unpack",
    "thicknesses",
    "sample_initial_parameters",
    "invert",
    "run_ensemble",
    "RunResult",
    "EnsembleResult",
    "save_geometry",
    "load_geometry",
]


# ---- the parameterisation ---------------------------------------------------


def parameter_names(tissue):
    """The recovered conductivities, in the order the parameter vector carries them.

    One entry per isotropic layer and two per anisotropic one, suffixed _T and _L.
    Derived from the tissue model rather than stated, so the vector and the labels
    on it cannot disagree: an isotropic tissue has one conductivity to recover, and
    naming two would invent a parameter the forward model does not have.
    """
    names = []
    for nm in tissue.layer_names:
        if nm in tissue.anisotropic_layers:
            names += [f"{nm}_T", f"{nm}_L"]
        else:
            names.append(nm)
    return tuple(names)


def sigma_indices(tissue):
    """`(transverse, longitudinal)` index tuples into the parameter vector, one
    entry per layer.

    An isotropic layer points both entries at the same element, enforcing the model's
    single-conductivity assumption and accumulating both directional sensitivities
    into that scalar.
    """
    iT, iL, k = [], [], 0
    for nm in tissue.layer_names:
        if nm in tissue.anisotropic_layers:
            iT.append(k)
            iL.append(k + 1)
            k += 2
        else:
            iT.append(k)
            iL.append(k)
            k += 1
    return tuple(iT), tuple(iL)


def unpack(theta, tissue):
    """Parameter vector -> `(sigma_T, sigma_L, interface radii)`.

    The leading entries are log-conductivities, so the exponential both enforces
    positivity and makes a step in the parameter a fixed *fraction* of the current
    value. This gives the optimiser a scale-invariant coordinate; it does not by
    itself guarantee equal conditioning of all conductivity directions.

    The trailing entries are softmax logits for the interfaces inside the outermost
    one, which `interfaces_from_logits` turns into radii that increase outwards by
    construction. The outermost radius is the tissue model's own and is not
    recovered: skin thickness is taken as known.

    Differentiable throughout, in both groups. `theta` is expected to require grad
    during an inversion; nothing here detaches it.
    """
    n_sigma = len(parameter_names(tissue))
    s = theta[:n_sigma].exp()
    iT, iL = sigma_indices(tissue)
    return (
        torch.stack([s[i] for i in iT]),
        torch.stack([s[i] for i in iL]),
        interfaces_from_logits(theta[n_sigma:], tissue.interfaces[-1]),
    )


def true_sigma(tissue):
    """The tissue model's own conductivities, laid out as `parameter_names`."""
    sT, sL = tissue.sigma_vectors()
    values = []
    for k, nm in enumerate(tissue.layer_names):
        values.append(sT[k])
        if nm in tissue.anisotropic_layers:
            values.append(sL[k])
    return torch.stack(values)


def true_parameters(tissue):
    """The parameter vector at the truth: `unpack` returns the tissue table.

    The inversion never sees this. It is the reference the relative errors are
    measured against, and the centre the starting points are drawn around.
    """
    return torch.cat(
        [
            torch.log(true_sigma(tissue)),
            logits_from_interfaces(tissue.interfaces),
        ]
    )


def thicknesses(interfaces, rim=0.5, scale=1.0):
    """Interface radii -> layer thicknesses, the outermost layer closed off at `rim`.

    A thickness is the anatomical quantity; a radius is a coordinate. `scale`
    converts the result into physical units -- pass `limb_radius / rim` for
    millimetres -- and is applied to the differences rather than to the radii, which
    is the same thing but keeps the caller from having to rescale `rim` too.
    """
    R = torch.as_tensor(interfaces).reshape(-1)
    edges = torch.cat(
        [torch.zeros_like(R[:1]), R, torch.full_like(R[:1], float(rim))]
    )
    return torch.diff(edges) * scale


# ---- the starting points ----------------------------------------------------


def sample_initial_parameters(
    tissue, n_runs, seed, log_spread=np.log(3.0), radius_spread=1.5 / 48.0
):
    """`(n_runs, n_parameters)` starting points drawn around the truth.

    Conductivities are drawn log-uniformly, `sigma = sigma_true * exp(U(-s, s))`,
    because sigma is a scale parameter: a factor of three up should be as likely as
    a factor of three down, and log is the coordinate the optimiser moves in.
    A uniform log-space distribution gives bounded multiplicative support, unlike
    a lognormal distribution with an unbounded tail.

    Interface radii are drawn uniformly, `R = R_true + U(-d, d)`, because a radius
    is a location and not a scale: what matters is how many millimetres, or grid
    cells, the guess is out by. The outermost radius is not perturbed; it is not
    recovered.

    Every parameter is drawn independently, including a muscle's transverse and
    longitudinal conductivities, so a small fraction of starts have the anisotropy
    the wrong way round. That is deliberate -- it tests the T/L split rather than
    assuming it -- and is worth reporting alongside the averages.

    Centred on the truth, which makes this a basin-of-attraction test and not a
    prior: it gives every parameter the same distribution of initial distance, which
    is what makes an averaged relative error interpretable.

    Parameters
    ----------
    tissue        : the model whose truth is the centre of the draws.
    n_runs        : how many starting points.
    seed          : seeds a generator private to this call, so the draws depend on
                    nothing else the session has done.
    log_spread    : half-width of the conductivity draw in log units. The default
                    ln 3 is a factor of three either way.
    radius_spread : half-width of the radius draw, in normalised radius units. The
                    default is 1.5 mm on a 24 mm limb, two radial cells at n = 65.

    Raises
    ------
    ValueError : if a draw puts the radii out of order or outside `(0, R_outer)`,
                 raised by `logits_from_interfaces`. At the default spread this
                 cannot happen; at a larger one it can, and it should stop the
                 experiment rather than be quietly resampled.
    """
    rng = np.random.default_rng(seed)
    rho_true = np.log(to_numpy(true_sigma(tissue)))
    R_true = np.asarray(tissue.interfaces[:-1], dtype=float)
    r_outer = tissue.interfaces[-1]

    starts = []
    for _ in range(int(n_runs)):
        rho = rho_true + rng.uniform(-log_spread, log_spread, rho_true.size)
        R = R_true + rng.uniform(-radius_spread, radius_spread, R_true.size)
        starts.append(
            torch.cat(
                [
                    torch.as_tensor(rho),
                    logits_from_interfaces(tuple(R) + (r_outer,)),
                ]
            )
        )
    return torch.stack(starts)


# ---- one run ----------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """What one inversion returned.

    Attributes
    ----------
    theta         : the recovered parameter vector.
    sigma         : its conductivities, ordered as `parameter_names`.
    interfaces    : its interface radii, outermost included and unchanged.
    loss_history  : the objective at every evaluation, in order. One entry per
                    evaluation and not per iteration, because that is what the
                    objective actually recorded and a line search makes the two
                    differ. It is the data misfit for an unregularised run and the
                    penalised total when `invert` was given a `tikhonov` weight --
                    what the optimiser saw, in both cases, rather than a
                    reconstruction of one part of it.
    geometry      : `(evaluation, sigma_T vector, interface radii)` at every
                    evaluation, aligned with `loss_history`. Seven scalars a
                    piece, detached from the graph, which is what makes recording
                    all of them affordable -- the alternative, a field per
                    snapshot, is 65^3 numbers. The spacing a figure is drawn at is
                    then chosen when it is drawn rather than when it is run.
    n_iterations,
    n_evaluations : totalled over all optimiser calls. The iteration count is
                    bounded by the global `max_iters` budget; evaluations are
                    managed against a separate target of four times that value.
    restart_at    : the index into `loss_history` of each restart, so how many
                    times the curvature memory was dropped is `n_restarts` and
                    where it happened is drawable. Empty means no restart was
                    launched, either because the first call met the stopping test
                    or because its remaining budget was exhausted.
    message,
    grad_max      : the optimiser's own exit report and max|g| at the returned
                    point. The message describes only the final SciPy call, while
                    max|g| provides a direct stationarity diagnostic for the
                    returned point.
    """

    theta: torch.Tensor
    sigma: torch.Tensor
    interfaces: torch.Tensor
    loss_history: np.ndarray
    geometry: tuple
    n_iterations: int
    n_evaluations: int
    restart_at: tuple
    message: str
    grad_max: float

    @property
    def n_restarts(self):
        """How many times the descent was resumed from a dropped curvature model.

        Derived rather than stored: it is the length of `restart_at`, and two
        fields that had to agree would eventually not.
        """
        return len(self.restart_at)


def invert(
    theta0,
    tissue,
    n,
    sources,
    observations,
    num_cycles=8,
    max_iters=1000,
    ftol=1e-16,
    gtol=1e-12,
    tikhonov=0.0,
    rho_ref=None,
):
    """One L-BFGS-B inversion from `theta0`.

    Each evaluation rebuilds the coefficient hierarchy with soft membership at the
    current radii and solves the forward problem once per source, so the same
    backward pass that returns dJ/dsigma also returns dJ/dR -- a shape derivative,
    obtained with no separate derivation.

    The objective and its gradient can be much smaller than one, so SciPy's generic
    defaults may terminate too early for this problem. The defaults here are
    tightened for that scale; callers should still inspect the returned message and
    `grad_max`.

    The solve is restarted to a fixed point rather than left to a single call --
    see the loop below for why, and for why that is still one inversion and not
    several stacked. `max_iters` is the global budget for the run, shared across
    the restarts, so accepted iterations share one total budget.

    `sources` and `observations` are parallel: observation k is the noisy field
    source k induced. The mask comes from each observation, so the electrodes are a
    property of the data rather than an argument that could disagree with it.

    Parameters
    ----------
    tikhonov : the weight on a penalty against the log-conductivities wandering
               from `rho_ref`, minimising J(theta) + (lambda/2)||rho - rho_ref||^2.
               Zero, the default, is the pure data misfit and leaves every
               expression below untouched.

               The penalty is on rho = log sigma and not on sigma, because that is
               the coordinate the optimiser carries and the one in which a factor
               of two up and a factor of two down are the same distance. A penalty
               on sigma would be a different prior, and a scale-dependent one: it
               would push bone at 0.02 and skin at 1.00 by amounts differing by a
               factor of fifty for the same relative error.

               The radii are deliberately not penalised. They are locations rather
               than scales, they enter through softmax logits where a norm has no
               natural meaning, and a geometry prior is a different claim from a
               conductivity prior -- so this regularises only the conductivity
               entries and leaves the geometry logits unpenalised.
    rho_ref  : the log-conductivities the penalty pulls towards, defaulting to
               those of `theta0`. That default makes the starting guess the prior,
               which is the usual reading of Tikhonov here: the penalty says the
               initial estimate was worth something, and lambda says how much.
               Ignored when `tikhonov` is zero.
    """
    if len(sources) != len(observations):
        raise ValueError(
            f"{len(sources)} sources against {len(observations)} observations; "
            "each source must have the field it induced"
        )

    n_sigma = len(parameter_names(tissue))
    if tikhonov < 0:
        raise ValueError(
            f"tikhonov weight must be non-negative, got {tikhonov}"
        )
    if rho_ref is None:
        rho_ref = torch.as_tensor(theta0)[:n_sigma].clone()
    rho_ref = torch.as_tensor(rho_ref, dtype=torch.float64).reshape(-1)
    if len(rho_ref) != n_sigma:
        raise ValueError(
            f"rho_ref has {len(rho_ref)} entries against {n_sigma} "
            "conductivities to regularise"
        )

    loss_history, geometry = [], []

    def objective(theta_np):
        """(fun, jac) at `theta_np`, the pair scipy asks for."""
        theta = torch.tensor(
            theta_np, dtype=torch.float64, requires_grad=True
        )
        sT_vec, sL_vec, R = unpack(theta, tissue)

        coeff = CoefficientHierarchy.from_vectors(
            tissue, n, sT_vec, sL_vec, interfaces=R
        )
        preds = [forward(I, coeff, num_cycles=num_cycles) for I in sources]

        loss = sum(
            0.5 * ((p - o.phi)[o.mask] ** 2).sum()
            for p, o in zip(preds, observations)
        )
        if tikhonov:
            loss = loss + 0.5 * tikhonov * (
                (theta[:n_sigma] - rho_ref) ** 2
            ).sum()
        loss.backward()

        # detached, or every snapshot would pin the multigrid graph that produced
        # it and the run would carry all of them at once
        geometry.append(
            (len(loss_history), sT_vec.detach().clone(), R.detach().clone())
        )
        loss_history.append(loss.item())
        return loss.item(), theta.grad.detach().numpy().copy()

    # ---- the restart loop ---------------------------------------------------
    # L-BFGS-B carries a limited-memory model of the curvature, built from its
    # last few correction pairs. Against the inexact gradient this problem
    # returns -- `probe_diagonal` runs under `no_grad` and the forward solve is a
    # fixed cycle budget, not a converged solve -- that model can go stale and the
    # line search may return ABNORMAL before the requested gradient tolerance is
    # met. Calling `minimize` again from `result.x` discards the accumulated
    # curvature pairs and starts with a fresh inverse-Hessian approximation.
    #
    # Restarting does not restart the run. The iterate is carried forward
    # untouched and only the curvature memory is dropped, so the parameter path
    # is unbroken. The operational stopping point is one at which another restart
    # accepts no step or fails to reduce the objective, unless a work budget is
    # exhausted first.
    #
    # The budget is global, not per restart. Each call is given what is left of
    # `max_iters` rather than a fresh `max_iters`, so the restarts share one
    # iteration budget. Function evaluations have their own shared budget below.
    x = to_numpy(torch.as_tensor(theta0)).astype(float)
    iters_left, evals_left = int(max_iters), 4 * int(max_iters)
    n_iterations = n_evaluations = 0
    previous_loss = np.inf
    restart_at = []

    while True:
        result = minimize(
            objective,
            x,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": iters_left,
                "maxfun": evals_left,
                "ftol": ftol,
                "gtol": gtol,
            },
        )
        n_iterations += int(result.nit)
        n_evaluations += int(result.nfev)
        iters_left -= int(result.nit)
        evals_left -= int(result.nfev)

        # Operational restart fixed point: the latest call accepted no iteration,
        # or it failed to improve on the previous call's terminal objective. This
        # does not prove a mathematical optimum; `message` and `grad_max` retain the
        # information needed to judge the returned point.
        #
        # Progress requires nit >= 1, so `iters_left` strictly decreases on every
        # pass that does not break, and the loop terminates on the budget even if
        # each restart buys an arbitrarily small decrease.
        converged_to_fixed_point = (
            result.nit == 0 or not result.fun < previous_loss
        )
        if converged_to_fixed_point or iters_left < 1 or evals_left < 1:
            break

        previous_loss = float(result.fun)
        x = result.x
        # where this restart falls in `loss_history`, so a figure can mark it.
        # The next call re-evaluates the point it is handed, so the entry at this
        # index repeats the one before it -- the misfit is flat across a boundary
        # by construction, so marking the boundary distinguishes that duplicate
        # evaluation from an ordinary flat segment.
        restart_at.append(len(loss_history))

    theta = torch.as_tensor(result.x)
    return RunResult(
        theta=theta,
        sigma=theta[:n_sigma].exp(),
        interfaces=interfaces_from_logits(
            theta[n_sigma:], tissue.interfaces[-1]
        ),
        loss_history=np.asarray(loss_history),
        geometry=tuple(geometry),
        n_iterations=n_iterations,
        n_evaluations=n_evaluations,
        restart_at=tuple(restart_at),
        message=str(result.message),
        # the infinity norm, because that is the norm L-BFGS-B's own pgtol test
        # uses: max|g| is the number `gtol` is a threshold on, and the 2-norm
        # this used to report is not comparable to it.
        grad_max=float(np.max(np.abs(result.jac))),
    )


# ---- the ensemble -----------------------------------------------------------


@dataclass(frozen=True)
class EnsembleResult:
    """The average of several inversions of the same data from different starts.

    Attributes
    ----------
    param_names       : the conductivity labels, ordered as the vectors below.
    layer_names       : the tissue labels, ordered as the thickness vectors.
    mean_sigma        : mean recovered conductivity over the runs.
    std_sigma         : the standard deviation of the runs about that mean, with
                        the n - 1 correction, so it estimates the spread of the
                        population of starting points rather than describing
                        these draws alone. It says how far a single run typically
                        lands from the ensemble's answer, which is the part of
                        the result the starting point is responsible for. It is
                        not the standard error of the mean, which is this divided
                        by sqrt(n) and answers a different question -- how well
                        the mean itself is pinned down. Either can be drawn as an
                        error bar, so a figure using one has to say which.
                        Undefined, and returned as nan, for a single run.
    mean_sigma_error  : mean over runs of |sigma_hat - sigma| / sigma, per
                        parameter. The mean of the errors, not the error of the
                        mean: the two differ, and it is the first that says how
                        wrong a typical run was.
    mean_interfaces   : mean recovered interface radii, the geometry the
                        parameterisation actually carries.
    mean_thickness,
    std_thickness     : the same two summaries for the layer thicknesses, in
                        normalised radius units. The outermost layer is fixed by
                        assumption, so its spread is identically zero.
    mean_thickness_error :
                        mean over runs of the relative thickness error, per layer.
                        The outermost layer's entry is identically zero: it is
                        fixed by assumption, not recovered.
    loss_histories    : one misfit history per run, in the order the runs were made.
    geometries        : one geometry history per run, same order and the same
                        triples `invert` records. Kept per-run rather than
                        averaged because a cross-section is a picture of one
                        inversion: averaging five geometries would draw a limb
                        that no run ever passed through.
    n_iterations,
    restart_at,
    messages,
    grad_max          : the per-run exit reports, so a caller can assess the points
                        included in the average. `unconverged` applies a simple
                        budget/stationarity screen; the raw reports remain available.
    """

    param_names: tuple
    layer_names: tuple
    mean_sigma: torch.Tensor
    std_sigma: torch.Tensor
    mean_sigma_error: torch.Tensor
    mean_interfaces: torch.Tensor
    mean_thickness: torch.Tensor
    std_thickness: torch.Tensor
    mean_thickness_error: torch.Tensor
    loss_histories: tuple
    geometries: tuple
    n_iterations: tuple
    restart_at: tuple
    messages: tuple
    grad_max: tuple

    @property
    def n_runs(self):
        return len(self.loss_histories)

    @property
    def n_restarts(self):
        """Restarts per run, in the order the runs were made."""
        return tuple(len(where) for where in self.restart_at)

    def unconverged(self, gtol):
        """Indices of runs flagged by budget termination or a large final gradient.

        A run is flagged when the final SciPy message begins with ``STOP`` or when
        max|g| exceeds `gtol`. Pass the same `gtol` used by `invert` if this screen
        should mirror the requested stationarity tolerance. The gradient is
        approximate because the probed smoother diagonal is detached and the
        forward solve uses a fixed cycle budget, so the raw values should be
        reported alongside this classification.
        """
        return tuple(
            k
            for k, (message, g) in enumerate(zip(self.messages, self.grad_max))
            if message.startswith("STOP") or g > gtol
        )


def _spread(values):
    """Standard deviation down the run axis, nan for a single run.

    Uses the n - 1 correction, so this is the sample standard deviation when the
    supplied runs are interpreted as draws from a starting-point distribution.
    PyTorch returns nan rather than zero at k = 1; one run does not estimate spread.
    """
    return values.std(0, correction=1)


def run_ensemble(starts, tissue, n, sources, observations, **kwargs):
    """Invert once from each row of `starts` and summarise the results.

    `kwargs` are passed to `invert` unchanged, so the explicit solver and optimiser
    settings are shared across runs. Assuming the called routines introduce no
    other varying state, each row of `starts` is then the intended difference.
    Runs may use different amounts of the common maximum budget because their
    restart sequences terminate at different times.

    The relative errors are reduced run by run rather than reconstructed from the
    means afterwards, because the mean of a ratio is not the ratio of the means and
    the first is the one that answers "how wrong is a typical run". The per-run
    values are held only long enough to take the two moments and are not returned:
    a single run's recovered numbers are a step towards the average, not a result.
    """
    sigma_ref = true_sigma(tissue)
    thick_ref = thicknesses(tissue.interfaces)

    sigma, radii, thick = [], [], []
    sigma_err = torch.zeros_like(sigma_ref)
    thick_err = torch.zeros_like(thick_ref)
    histories, geometries, iterations = [], [], []
    restart_at, messages, grad_max = [], [], []

    for theta0 in torch.as_tensor(starts):
        run = invert(theta0, tissue, n, sources, observations, **kwargs)
        sigma.append(run.sigma)
        radii.append(run.interfaces)
        thick.append(thicknesses(run.interfaces))
        sigma_err += (run.sigma - sigma_ref).abs() / sigma_ref
        thick_err += (thick[-1] - thick_ref).abs() / thick_ref
        histories.append(run.loss_history)
        geometries.append(run.geometry)
        iterations.append(run.n_iterations)
        restart_at.append(run.restart_at)
        messages.append(run.message)
        grad_max.append(run.grad_max)

    k = len(histories)
    sigma = torch.stack(sigma)
    thick = torch.stack(thick)
    return EnsembleResult(
        param_names=parameter_names(tissue),
        layer_names=tuple(tissue.layer_names),
        mean_sigma=sigma.mean(0),
        std_sigma=_spread(sigma),
        mean_sigma_error=sigma_err / k,
        mean_interfaces=torch.stack(radii).mean(0),
        mean_thickness=thick.mean(0),
        std_thickness=_spread(thick),
        mean_thickness_error=thick_err / k,
        loss_histories=tuple(histories),
        geometries=tuple(geometries),
        n_iterations=tuple(iterations),
        restart_at=tuple(restart_at),
        messages=tuple(messages),
        grad_max=tuple(grad_max),
    )


# ---- persistence ------------------------------------------------------------

# These helpers persist the compact geometry history used by cross-section figures,
# allowing panel spacing and layout to change without rerunning the inversion.
# Persistence of other run outputs is left to the caller.
#
# A small vector per evaluation is what makes this worth doing: a few hundred
# snapshots is a few tens of kilobytes, so a run's whole geometry history is stored
# rather than a chosen subset of it, and which evaluations a figure draws stays a
# decision made when it is drawn.

_GEOMETRY_ARRAYS = ("evaluation", "sigma_T", "interfaces")
_METADATA_PREFIX = "meta_"


def save_geometry(path, snapshots, **metadata):
    """Write one run's geometry history to `path` as an npz, and return the path.

    `snapshots` is what `RunResult.geometry` carries: `(evaluation, sigma_T vector,
    interface radii)` triples. They are stored columnwise -- one array of evaluation
    indices, one of stacked sigma_T vectors, one of stacked radii -- rather than as
    a triple per snapshot, because that is one array read per field on the way back
    in instead of one per iterate.

    `metadata` is any extra scalars the caller wants to travel with the file, keyed
    under a prefix so a metadata name can never collide with an array name. `n` is
    the one worth passing: the grid the figure draws on is rebuilt from it, and a
    geometry file that does not say which resolution it came from can be drawn on
    the wrong one without anything raising.

    The format is NumPy ``npz`` rather than pickle or ``torch.save`` because the
    contents are ordinary numeric arrays and metadata scalars; no live Python
    objects need to be reconstructed when the file is loaded.
    """
    if len(snapshots) == 0:
        raise ValueError("no snapshots to save: a run with no iterates")

    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = {
        "evaluation": np.asarray(
            [int(it) for it, _, _ in snapshots], dtype=np.int64
        ),
        "sigma_T": np.stack([to_numpy(sT) for _, sT, _ in snapshots]),
        "interfaces": np.stack([to_numpy(R) for _, _, R in snapshots]),
    }
    named = {_METADATA_PREFIX + k: np.asarray(v) for k, v in metadata.items()}
    np.savez_compressed(path, **columns, **named)
    return path


def load_geometry(path):
    """Read back what `save_geometry` wrote, as `(snapshots, metadata)`.

    The snapshots come back in exactly the form `invert` produced them and
    `viz.plot_geometry_panels` consumes -- a tuple of `(int, tensor, tensor)` --
    so a figure drawn from a file and one drawn from a live run are the same call
    with the same arguments. That is the point of the round trip: the spacing, the
    slice and the figure size stay decisions made at drawing time, and only the
    inversion is expensive enough to want doing once.
    """
    with np.load(Path(path).with_suffix(".npz")) as f:
        missing = [k for k in _GEOMETRY_ARRAYS if k not in f]
        if missing:
            raise ValueError(
                f"{path} is missing {missing}; it is not a geometry file"
            )
        evaluation, sigma_T, interfaces = (f[k] for k in _GEOMETRY_ARRAYS)
        metadata = {
            k[len(_METADATA_PREFIX) :]: (v.item() if v.ndim == 0 else v)
            for k, v in f.items()
            if k.startswith(_METADATA_PREFIX)
        }

    snapshots = tuple(
        (int(it), torch.as_tensor(sT), torch.as_tensor(R))
        for it, sT, R in zip(evaluation, sigma_T, interfaces)
    )
    return snapshots, metadata
