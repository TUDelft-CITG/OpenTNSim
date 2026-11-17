"""
Utilities for OpenTNSim

This file also includes the networkx read shapefile functions that used to be in networkx.
These originate from:
https://github.com/networkx/networkx/blob/9256ef670730b741105a9264197353004bd6719f/networkx/readwrite/nx_shp.py

Generates a networkx.DiGraph from point and line shapefiles.

"The Esri Shapefile or simply a shapefile is a popular geospatial vector
data format for geographic information systems software. It is developed
and regulated by Esri as a (mostly) open specification for data
interoperability among Esri and other software products."
See https://en.wikipedia.org/wiki/Shapefile for additional information.
"""

from typing import Any, Dict, List, Tuple, Type

# packkage(s) for documentation, debugging, saving and loading
import pathlib
import inspect
from IPython.display import display

# time libraries
import datetime

# data libraries
import numpy as np
import pandas as pd
import itertools
import scipy.stats as stats
import ast
from scipy.optimize import brentq
from scipy.special import gamma

# OpenTNSim
import opentnsim


def inherit_docstring(cls):
    for name, func in cls.__dict__.items():
        if callable(func) and not func.__doc__:
            parent_func = getattr(super(cls, cls), name, None)
            if parent_func:
                func.__doc__ = parent_func.__doc__
    return cls


def find_notebook_path():
    """Lookup the path where the notebooks are located. Returns a pathlib.Path object."""
    opentnsim_path = pathlib.Path(opentnsim.__file__)
    # check if the path looks normal
    assert "opentnsim" in str(opentnsim_path), "we can't find the opentnsim path: {opentnsim_path} (opentnsim not in path name)"
    # src_dir/opentnsim/__init__.py -> ../.. -> src_dir
    notebook_path = opentnsim_path.parent.parent / "notebooks"
    return notebook_path


def time_to_numpy(t_start):
    """Convert time to np.datetime64

    Parameters
    ----------
    t_start : float, datetime.datetime, pd.Timestamp
        the time to be converted
    Returns
    -------
    t_start : np.datetime64
        the converted time
    """
    if isinstance(t_start, float):
        t_start = np.datetime64(datetime.datetime.fromtimestamp(t_start))
    elif isinstance(t_start, datetime.datetime):
        t_start = np.datetime64(t_start)
    elif isinstance(t_start, pd.Timestamp):
        t_start = np.array([t_start], dtype=np.datetime64)[0]
    return t_start


def inspect_object(Class, candidate_kwargs=None, show_parameter_table=True):

    documenting = not candidate_kwargs

    def highlight_status(row):
        colors = {
            "missing": "#ffcccc",
            "missing (one alternative)": "#ffcccc",
            "optional (unused)": "#ffe5b4",
            "optional (used)": "#e8f5e9",
            "required": "#6fa6cf",
            "required (one alternative)": "#6fa6cf",
            "optional": "#afcee8",
            "added": "#c8e6c9",
        }
        color = colors.get(row["status"], "#ffffff")
        return [f"background-color: {color}"] * len(row)

    candidate_kwargs = candidate_kwargs or {}

    # Collect alternative requirement groups from the class hierarchy.
    # Each tuple is a group of parameters that must all be supplied.
    # At least one group must be satisfied.
    alternative_requirements = []

    for cls in Class.mro():
        alternative_requirements.extend(
            cls.__dict__.get("__parameter_requirements__", [])
        )

    # Collect constructor parameters.
    rows = []

    for cls in Class.mro():
        if cls is object or "__init__" not in cls.__dict__:
            continue

        sig = inspect.signature(cls.__init__)

        for name, param in sig.parameters.items():
            if name == "self":
                continue

            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            rows.append({
                "parameter": name,
                "class": cls.__name__,
                "required": param.default is inspect.Parameter.empty,
                "provided": name in candidate_kwargs,
            })

    df = pd.DataFrame(rows).drop_duplicates("parameter")

    if df.empty:
        if show_parameter_table:
            display(df)
            return
        return df, False

    # Check alternative requirements.
    provided = set(candidate_kwargs)

    alternatives_satisfied = (
        not alternative_requirements
        or any(
            all(parameter in provided for parameter in group)
            for group in alternative_requirements
        )
    )

    # Add a row for an unsatisfied alternative requirement.
    df["status"] = " "
    if alternative_requirements and not alternatives_satisfied:
        alternatives_label = " OR ".join(
            " + ".join(group)
            for group in alternative_requirements
        )

        df = pd.concat(
            [
                df,
                pd.DataFrame([{
                    "parameter": f"{alternatives_label}",
                    "class": Class.__name__,
                    "required": True,
                    "provided": False,
                    "status": "combination",
                }]),
            ],
            ignore_index=True,
        )

    # Assign statuses based on the mode.
    df["missing"] = df["required"] & ~df["provided"]
    df.loc[
        df["missing"] & (df["status"] != "combination"), "status"
    ] = "missing"
    df.loc[
        df["required"] & df["provided"], "status"
    ] = "added"
    df.loc[
        ~df["required"] & df["provided"], "status"
    ] = "optional (used)"
    df.loc[
        ~df["required"] & ~df["provided"], "status"
    ] = "optional (unused)"

    # Apply alternative requirement status.
    combination_mask = df["status"].str.startswith(
        "combi"
    )

    if alternative_requirements:
        if documenting:
            df.loc[
                combination_mask, "status"
            ] = "required (one alternative)"
        elif not alternatives_satisfied:
            df.loc[
                combination_mask, "status"
            ] = "missing (one alternative)"
        else:
            # The combination requirement is satisfied.
            df.loc[combination_mask, "status"] = "added"

    # In documentation mode, show required/optional parameters.
    if documenting:
        df.loc[df["status"] == "missing", "status"] = "required"
        df.loc[
            df["status"] == "optional (unused)", "status"
        ] = "optional"

    # In validation mode, report missing individual and combination requirements.
    missing_parameters = (
        not documenting
        and (
            df["status"].isin(
                ["missing", "requirement missing"]
            ).any()
        )
    )

    df = df.sort_values(
        by=["class", "parameter"],
        ascending=[True, True],
    ).reset_index(drop=True)

    df = df.drop(
        ["missing", "required", "provided"],
        axis=1,
    )

    if documenting:
        status_order = {
            "required": 0,
            "required (one alternative)": 1,
            "optional": 2,
        }
    else:
        status_order = {
            "added": 0,
            "optional (used)": 1,
            "missing": 2,
            "requirement missing": 3,
            "optional (unused)": 4,
        }

    df["_sort_order"] = df["status"].map(status_order).fillna(99)

    df = (
        df.sort_values(
            by=["_sort_order", "class", "parameter"],
            ascending=[True, True, True],
        )
        .drop(columns="_sort_order")
        .reset_index(drop=True)
    )

    if show_parameter_table:
        display(df.style.apply(highlight_status, axis=1))
    else:
        return df, missing_parameters


def check_class_is_vesselclass(vessel_class):
    required_mixins = ["Identifiable", "Movable"]
    present_mixins = [cls.__name__ for cls in vessel_class.mro()]
    missing_mixins = [m for m in required_mixins if m not in present_mixins]

    if missing_mixins:
        raise TypeError(f"vessel_class must include the following mixins: {missing_mixins}")


def get_all_init_params(vessel_class):
    """
    Return a set of all parameters from __init__ in the MRO
    (excluding self, *args, **kwargs)
    """
    allowed = set()
    for cls in vessel_class.mro():
        if cls is object:
            continue
        if "__init__" not in cls.__dict__:
            continue
        sig = inspect.signature(cls.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            allowed.add(name)
    return allowed


DISTRIBUTIONS = {
    "triangular": {
        "parameters": ["low", "mode", "high"],
        "description": "Triangular distribution",
        "example": "low=60, mode=80, high=100",
        "factory": lambda p: stats.triang(
            c=(p["mode"] - p["low"]) / (p["high"] - p["low"]),
            loc=p["low"],
            scale=p["high"] - p["low"],
        ),
        "validate": lambda p: (
            p["low"] < p["high"]
            and p["low"] <= p["mode"] <= p["high"]
        ),
    },

    "uniform": {
        "parameters": ["low", "high"],
        "description": "Uniform distribution",
        "example": "low=60, high=100",
        "factory": lambda p: stats.uniform(
            loc=p["low"],
            scale=p["high"] - p["low"],
        ),
        "validate": lambda p: p["low"] < p["high"],
    },

    "normal": {
        "parameters": ["mean", "std"],
        "description": "Normal distribution",
        "example": "mean=80, std=10",
        "factory": lambda p: stats.norm(
            loc=p["mean"],
            scale=p["std"],
        ),
        "validate": lambda p: p["std"] > 0,
    },

    "lognormal": {
        "parameters": ["mean", "sigma"],
        "description": "Lognormal distribution",
        "example": "mean=4.3, sigma=0.2",
        "factory": lambda p: stats.lognorm(
            s=p["sigma"],
            scale=np.exp(p["mean"]),
        ),
        "validate": lambda p: p["sigma"] > 0,
    },

    "constant": {
        "parameters": ["value"],
        "description": "Constant value",
        "example": "value=80",
        "factory": lambda p: stats.rv_discrete(
            values=([p["value"]], [1.0])
        ),
        "validate": lambda p: True,
    },

    "gamma": {
        "parameters": ["mean", "cv"],
        "description": "Gamma distribution",
        "example": "mean=10, cv=0.5",
        "factory": lambda p: stats.gamma(
            a=1 / p["cv"]**2,
            scale=p["mean"] * p["cv"]**2,
        ),
        "validate": lambda p: (
            p["mean"] > 0
            and p["cv"] > 0
        ),
    },
       
    "discrete": {
        "parameters": ["values", "probabilities"],
        "description": "Discrete probability distribution",
        "example": "values=['small_cargo', 'general_cargo'], probabilities=[0.4, 0.6]",
        "factory": lambda p: stats.rv_discrete(
            values=(
                np.arange(len(p["values"])),
                p["probabilities"],
            )
        ),
        "validate": lambda p: (
            len(p["values"]) > 0
            and len(p["values"]) == len(p["probabilities"])
            and len(set(p["values"])) == len(p["values"])
            and all(
                np.isfinite(probability) and probability >= 0
                for probability in p["probabilities"]
            )
            and np.isclose(sum(p["probabilities"]), 1.0)
        ),
    },
    "weibull": {
        "parameters": ["mean", "cv"],
        "description": "Weibull distribution (mean and coefficient of variation)",
        "example": "mean=17.7, cv=0.523",
        "factory": lambda p: stats.weibull_min(
            c=(
                brentq(
                    lambda k: (
                        np.sqrt(
                            gamma(1 + 2/k) /
                            gamma(1 + 1/k)**2 - 1
                        ) - p["cv"]
                    ),
                    0.1, 100
                )
            ),
            scale=(
                p["mean"] /
                gamma(
                    1 + 1 / brentq(
                        lambda k: (
                            np.sqrt(
                                gamma(1 + 2/k) /
                                gamma(1 + 1/k)**2 - 1
                            ) - p["cv"]
                        ),
                        0.1, 100
                    )
                )
            ),
        ),
        "validate": lambda p: (
            p["mean"] > 0
            and p["cv"] > 0
        ),
    },
}


def list_distributions():
    """Display supported distributions and required parameters."""

    return pd.DataFrame([
        {
            "distribution": name,
            "description": spec["description"],
            "required_parameters": ", ".join(spec["parameters"]),
            "example": spec["example"],
        }
        for name, spec in DISTRIBUTIONS.items()
    ])



class Distribution:

    def __init__(
        self,
        name,
        parameters,
        scipy_distribution,
        seed=None,
        offset=0.0,
        multiplier=1.0,
    ):
        self.name = name
        self.parameters = parameters
        self.scipy_distribution = scipy_distribution
        self.rng = np.random.default_rng(seed)

        self.offset = offset
        self.multiplier = multiplier

    def rvs(self, *args, **kwargs):
        """Generate random samples."""
        samples = self.scipy_distribution.rvs(
            *args,
            random_state=self.rng,
            **kwargs,
        )

        return self.offset + self.multiplier * samples

    def ppf(self, q):
        """Percent point function (inverse CDF)."""
        if self.multiplier >= 0:
            return (
                self.offset
                + self.multiplier * self.scipy_distribution.ppf(q)
            )

        return (
            self.offset
            + self.multiplier * self.scipy_distribution.ppf(
                1 - np.asarray(q)
            )
        )

    def cdf(self, x):
        """Cumulative distribution function."""
        if self.multiplier == 0:
            return np.where(
                np.asarray(x) < self.offset, 0.0, 1.0
            )

        z = (np.asarray(x) - self.offset) / self.multiplier

        if self.multiplier > 0:
            return self.scipy_distribution.cdf(z)

        return self.scipy_distribution.sf(z)

    def pdf(self, x):
        """Probability density function."""
        if self.multiplier == 0:
            raise ValueError(
                "PDF is undefined for a constant distribution."
            )

        z = (np.asarray(x) - self.offset) / self.multiplier

        return (
            self.scipy_distribution.pdf(z)
            / abs(self.multiplier)
        )

    def mean(self):
        """Expected value."""
        return (
            self.offset
            + self.multiplier * self.scipy_distribution.mean()
        )

    def var(self):
        """Variance."""
        return (
            self.multiplier**2
            * self.scipy_distribution.var()
        )

    def std(self):
        """Standard deviation."""
        return abs(self.multiplier) * self.scipy_distribution.std()

    def __repr__(self):
        params = ", ".join(
            f"{key}={value}"
            for key, value in self.parameters.items()
        )

        transformation = (
            f", offset={self.offset}, scale={self.multiplier}"
        )

        return f"{self.name}({params}{transformation})"


def create_distribution(
    distribution,
    *,
    seed=None,
    offset=0.0,
    scale=1.0,
    **parameters,
):
    """Create a validated and optionally transformed distribution."""

    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported distribution '{distribution}'."
        )

    specification = DISTRIBUTIONS[distribution]

    required = set(specification["parameters"])
    supplied = set(parameters)

    missing = required - supplied
    unexpected = supplied - required

    if missing or unexpected:
        raise ValueError(
            f"Invalid parameters for '{distribution}'. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )

    # Validate distribution parameters
    if distribution == "discrete":

        values = parameters["values"]
        probabilities = parameters["probabilities"]

        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(
                "Parameter 'values' must be a sequence."
            )

        if not isinstance(probabilities, (list, tuple, np.ndarray)):
            raise ValueError(
                "Parameter 'probabilities' must be a numerical sequence."
            )

        if not all(
            isinstance(v, (str, int, float, bool, np.integer,
                           np.floating, np.bool_))
            and (
                not isinstance(v, (float, np.floating))
                or np.isfinite(v)
            )
            for v in values
        ):
            raise ValueError(
                "Discrete values must be strings or finite numerical values."
            )

        if not all(
            isinstance(p, (int, float, np.integer, np.floating))
            and not isinstance(p, (bool, np.bool_))
            and np.isfinite(p)
            for p in probabilities
        ):
            raise ValueError(
                "Probabilities must contain finite numerical values."
            )

    else:
        # Other distributions require finite numerical parameters.
        for key, value in parameters.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value,
                    (int, float, np.integer, np.floating)
                )
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"Parameter '{key}' must be a finite real number."
                )

    if not specification["validate"](parameters):
        raise ValueError(
            f"Invalid parameter values for '{distribution}': "
            f"{parameters}"
        )

    if not np.isfinite(offset) or not np.isfinite(scale):
        raise ValueError("Offset and scale must be finite.")

    scipy_distribution = specification["factory"](parameters)

    return Distribution(
        name=distribution,
        parameters=parameters.copy(),
        scipy_distribution=scipy_distribution,
        seed=seed,
        offset=offset,
        multiplier=scale,
    )


def validate_generation_inputs(
    env,
    origin,
    destination,
    fleet_data,
    fleet_composition,
    arrival_distribution,
    number_of_vessels,
    correlation_matrices=None,
):
    """Validate vessel configuration and generation inputs."""

    # 1. Validate general inputs
    if (
        isinstance(number_of_vessels, (bool, np.bool_))
        or not isinstance(number_of_vessels, (int, np.integer))
        or number_of_vessels < 0
    ):
        raise ValueError(
            "number_of_vessels must be a non-negative integer."
        )

    if not isinstance(arrival_distribution, Distribution):
        raise TypeError(
            "arrival_distribution must be a Distribution object."
        )

    # 2. Validate DataFrame structure
    required_columns = [
        "vessel_class",
        "vessel_object",
    ]

    missing_columns = set(required_columns) - set(fleet_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing DataFrame columns: {sorted(missing_columns)}"
        )

    if fleet_data.empty:
        raise ValueError("fleet_data cannot be empty.")

    if fleet_data["vessel_class"].duplicated().any():
        raise ValueError("Each vessel class must appear only once.")

    if fleet_data[required_columns].isnull().any().any():
        raise ValueError(
            "Required vessel configuration values cannot be null."
        )

    # 3. Validate vessel object parameters
    external_parameters = {
        "env": env,
        "origin": origin,
        "destination": destination,
    }

    available_parameters = set(fleet_data.columns)

    available_parameters.update(
        key for key, value in external_parameters.items()
        if value is not None
    )

    for vessel_data in fleet_data.itertuples(index=False):

        vessel_object = vessel_data.vessel_object

        vessel_parameters, _ = inspect_object(
            vessel_object,
            show_parameter_table=False,
        )

        required_parameters = vessel_parameters.loc[
            vessel_parameters["status"] == "required",
            "parameter",
        ].tolist()

        missing_parameters = (
            set(required_parameters) - available_parameters
        )

        if missing_parameters:
            raise ValueError(
                f"Missing required parameters for "
                f"{vessel_data.vessel_class}: "
                f"{sorted(missing_parameters)}"
            )

    # 4. Validate fleet composition structure
    composition_columns = [
        "vessel_class",
        "fleet_probability",
    ]

    missing_columns = (
        set(composition_columns) - set(fleet_composition.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing fleet composition columns: "
            f"{sorted(missing_columns)}"
        )

    if fleet_composition.empty:
        raise ValueError("fleet_composition cannot be empty.")

    if fleet_composition[composition_columns].isnull().any().any():
        raise ValueError(
            "Fleet composition values cannot be null."
        )

    if fleet_composition["vessel_class"].duplicated().any():
        raise ValueError(
            "Each vessel class must appear only once "
            "in fleet composition."
        )

    # 5. Validate vessel classes
    composition_classes = set(fleet_composition["vessel_class"])
    fleet_classes = set(fleet_data["vessel_class"])

    missing_classes = fleet_classes - composition_classes
    unexpected_classes = composition_classes - fleet_classes

    if missing_classes or unexpected_classes:
        raise ValueError(
            "Fleet composition does not match fleet data. "
            f"Missing classes: {sorted(missing_classes)}. "
            f"Unexpected classes: {sorted(unexpected_classes)}."
        )

    # 6. Validate fleet probabilities
    probabilities = fleet_composition[
        "fleet_probability"
    ].to_numpy(dtype=float)

    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError(
            "Fleet probabilities must be finite, "
            "non-negative and sum to one."
        )

    # 7. Validate correlation matrices.
    validate_correlation_matrices(
        correlation_matrices=correlation_matrices,
        fleet_data=fleet_data,
    )

    return True



def validate_correlation_matrices(
    correlation_matrices,
    fleet_data,
):
    """Validate correlation matrices for the vessel classes."""

    if correlation_matrices is None:
        return True

    if not isinstance(correlation_matrices, dict):
        raise TypeError(
            "correlation_matrices must be a dictionary."
        )

    fleet_classes = set(fleet_data["vessel_class"])

    # Check that correlation matrices refer to existing vessel classes.
    unknown_classes = set(correlation_matrices) - fleet_classes

    if unknown_classes:
        raise ValueError(
            "Correlation matrices contain unknown vessel classes: "
            f"{sorted(unknown_classes)}."
        )

    for vessel_class, matrix in correlation_matrices.items():

        if not isinstance(matrix, pd.DataFrame):
            raise TypeError(
                f"Correlation matrix for '{vessel_class}' "
                "must be a pandas DataFrame."
            )

        # Check for duplicate variable names.
        if matrix.index.has_duplicates:
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "contains duplicate index names."
            )

        if matrix.columns.has_duplicates:
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "contains duplicate column names."
            )

        # Check that index and columns contain the same variables.
        if (
            len(matrix.index) != len(matrix.columns)
            or set(matrix.index) != set(matrix.columns)
        ):
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "must have identical index and column names."
            )

        # Ensure the columns follow the same order as the index.
        matrix = matrix.loc[matrix.index, matrix.index]

        # Check numerical values.
        try:
            values = matrix.to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "must contain numerical values."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "must contain finite values."
            )

        # Check diagonal.
        if not np.allclose(np.diag(values), 1.0):
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "must have ones on its diagonal."
            )

        # Check symmetry.
        if not np.allclose(values, values.T):
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "must be symmetric."
            )

        # Check correlation bounds.
        if np.any(np.abs(values) > 1):
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "must contain values between -1 and 1."
            )

        eigenvalues = np.linalg.eigvalsh(values)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"Correlation matrix for '{vessel_class}' "
                "must be positive semidefinite."
            )

    return True


def generate_arrival_times(
    number_of_vessels,
    start_time,
    arrival_distribution,
):
    """
    Generate vessel arrival times using an interarrival-time
    probability distribution.
    """

    if number_of_vessels == 0:
        return np.array([])

    interarrival_times = arrival_distribution.rvs(
        size=number_of_vessels,
    )

    return [start_time + pd.Timedelta(minutes = time) for time in np.cumsum(interarrival_times)]


def generate_vessels_from_distribution(
    env,
    fleet_data,
    fleet_composition,
    arrival_distribution,
    number_of_vessels,
    origin=None,
    destination=None,
    route=None,
    fleet_seed=None,
    start_time=None,
    correlation_matrices=None,
):
    """
    Generate vessels using a single vessel configuration DataFrame.
    """
    if (origin is None or destination is None) and route is None:
        raise ValueError('Either an origin/destination node pair or route should be defined')
    if route is not None:
        origin = route[0]
        destination = route[-1]

    # Validate inputs.
    validate_generation_inputs(
        env=env,
        origin=origin,
        destination=destination,
        fleet_data=fleet_data,
        fleet_composition=fleet_composition,
        arrival_distribution=arrival_distribution,
        number_of_vessels=number_of_vessels,
        correlation_matrices=correlation_matrices,
    )

    if number_of_vessels == 0:
        return []

    # Determine starting time.
    if start_time is None:
        start_time = env.simulation_start

    # Generate arrival times.
    arrival_times = generate_arrival_times(
        number_of_vessels=number_of_vessels,
        start_time=start_time,
        arrival_distribution=arrival_distribution,
    )

    # Determine vessel numbering.
    earlier_vessels = len(getattr(env, "vessels", []))

    vessels = []

    fleet_distribution = create_distribution(
        "discrete",
        values=fleet_composition["vessel_class"].tolist(),
        probabilities=fleet_composition["fleet_probability"].tolist(),
        seed=fleet_seed,
    )

    # Generate vessels.
    for i, arrival_time in enumerate(arrival_times):

        # Select vessel class.
        row = select_vessel_class(
            fleet_data=fleet_data,
            fleet_distribution=fleet_distribution, 
        )

        # Generate vessel.
        vessel = generate_single_vessel(
            env=env,
            row=row,
            origin=origin,
            destination=destination,
            arrival_time=arrival_time,
            vessel_name=f"Vessel {earlier_vessels + i}",
            correlation_matrices = correlation_matrices,
        )

        vessels.append(vessel)

    return vessels


def sample_correlated_parameters(parameters, correlation_matrix):
    """
    Sample vessel parameters using a Gaussian copula.

    Parameters not included in the correlation matrix are sampled
    independently using their existing distribution objects.
    """

    sampled_parameters = {}
    distributions = {}

    # Separate distributions, callables and fixed values
    for parameter, value in parameters.items():

        distribution = _get_distribution(value)

        if distribution is not None:
            distributions[parameter] = distribution

        elif callable(value):
            sampled_parameters[parameter] = value()

        else:
            sampled_parameters[parameter] = value

    # No correlation matrix: use the original independent sampling
    if correlation_matrix is None:
        for parameter, distribution in distributions.items():
            sampled_parameters[parameter] = distribution.rvs()

        return sampled_parameters

    # Identify stochastic parameters included in the matrix
    correlated_parameters = [
        parameter
        for parameter in correlation_matrix.index
        if parameter in distributions
    ]

    # Sample parameters outside the matrix independently
    independent_parameters = [
        parameter
        for parameter in distributions
        if parameter not in correlated_parameters
    ]

    for parameter in independent_parameters:
        sampled_parameters[parameter] = distributions[parameter].rvs()

    if not correlated_parameters:
        return sampled_parameters

    # Extract and reorder the relevant correlation matrix
    matrix = correlation_matrix.loc[
        correlated_parameters, correlated_parameters
    ].to_numpy(dtype=float)

    # Eigenvalue decomposition also handles positive-semidefinite matrices
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    # Generate independent standard normal values using the RNG
    # already stored in each parameter's distribution
    independent_normals = np.array([
        _get_stored_rng(distributions[parameter]).standard_normal()
        for parameter in correlated_parameters
    ])

    # Introduce the requested correlations
    correlated_normals = factor @ independent_normals

    # Convert correlated normal values into uniform values
    uniform_values = stats.norm.cdf(correlated_normals)

    # Avoid numerical problems at the boundaries
    epsilon = np.finfo(float).eps
    uniform_values = np.clip(
        uniform_values,
        epsilon,
        1.0 - epsilon
    )

    # Apply each parameter's original marginal distribution
    for parameter, uniform_value in zip(
        correlated_parameters, uniform_values
    ):

        distribution = _get_scipy_distribution(
            distributions[parameter]
        )

        sampled_parameters[parameter] = distribution.ppf(
            uniform_value
        )

    return sampled_parameters


def _get_distribution(value):
    """Convert a parameter specification into a distribution object."""

    if isinstance(value, str):
        expression = ast.parse(value, mode="eval").body

        if not isinstance(expression, ast.Call) or not isinstance(
            expression.func, ast.Name
        ):
            raise ValueError(f"Invalid distribution specification: {value}")

        distribution_name = expression.func.id

        distribution_parameters = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in expression.keywords
            if keyword.arg is not None
        }

        return create_distribution(
            distribution_name,
            **distribution_parameters
        )

    if hasattr(value, "rvs"):
        return value

    return None


def _get_scipy_distribution(distribution):
    """Get the underlying SciPy distribution for inverse-CDF sampling."""

    if hasattr(distribution, "ppf"):
        return distribution

    if hasattr(distribution, "distribution"):
        underlying = distribution.distribution

        if hasattr(underlying, "ppf"):
            return underlying

    raise TypeError(
        "The distribution must expose a SciPy distribution with a ppf method."
    )


def _get_stored_rng(distribution):
    """Retrieve the random generator already stored in a Distribution."""

    rng = getattr(distribution, "rng", None)

    if rng is None:
        raise TypeError(
            "Correlated sampling requires a distribution with a stored "
            "random generator (distribution.rng)."
        )

    return rng


def generate_single_vessel(
    env,
    row,
    origin,
    destination,
    vessel_name,
    arrival_time=None,
    show_unused_optional_parameters=False,
    correlation_matrices=None,
):
    """Generate a vessel from a selected fleet composition row."""

    vessel_object = row["vessel_object"]

    # Get the correlation matrix for this vessel class
    correlation_matrix = None

    if correlation_matrices is not None:

        if isinstance(correlation_matrices, dict):
            vessel_class = row["vessel_class"]
            correlation_matrix = correlation_matrices.get(vessel_class)

        else:
            # Also allow a single DataFrame to be supplied directly
            correlation_matrix = correlation_matrices

    # Collect parameter specifications
    parameters = {}
    for parameter, value in row.items():

        if isinstance(value, Distribution):
            parameters[parameter] = value.rvs()

        else:
            parameters[parameter] = value

    parameters['name'] = vessel_name
    if arrival_time is not None:
        parameters['arrival_time'] = arrival_time

    return create_vessel(
        env=env,
        vessel_obj=vessel_object,
        origin=origin,
        destination=destination,
        show_unused_optional_parameters=show_unused_optional_parameters,
        **parameters,
    )


def select_vessel_class(
    fleet_data,
    fleet_distribution,
):
    """Select a vessel class and return its DataFrame row."""
    index = int(fleet_distribution.rvs())
    vessel_class = fleet_distribution.parameters["values"][index]

    selected = fleet_data[
        fleet_data["vessel_class"] == vessel_class
    ]

    if selected.empty:
        raise ValueError(
            f"Vessel class '{vessel_class}' not found."
        )

    return selected.iloc[0]


def highlight_row_if_required(row):
    color = 'background-color: #c6efce' if bool(row['required']) else ''
    return [color] * len(row)


def describe_inits_by_mro(cls: Type[Any]) -> Dict[str, Any]:
    """
    Inspect each base in MRO and capture its __init__ parameters (excluding *args/**kwargs).
    Returns a dict with 'per_base' (mixin -> rows) and 'union' (first-seen param in MRO).
    """
    per_base: Dict[str, List[Dict[str, Any]]] = {}
    union: Dict[str, Dict[str, Any]] = {}

    for base in cls.__mro__:
        if base is object:
            continue
        init = getattr(base, "__init__", object.__init__)
        if init is object.__init__:
            continue

        sig = inspect.signature(init)
        rows: List[Dict[str, Any]] = []  # <-- fixed bracket

        for pname, p in sig.parameters.items():
            if pname == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):  # skip *args/**kwargs
                continue

            rows.append({
                "name": pname,
                "required": (p.default is inspect._empty),
                "default": None if p.default is inspect._empty else p.default,
                "annotation": None if p.annotation is inspect._empty else p.annotation,
                "kind": p.kind.name,
                "declared_in": base.__name__,
            })

            # Union: first occurrence in MRO wins if duplicates
            if pname not in union:
                union[pname] = rows[-1].copy()

        per_base[base.__name__] = rows

    return {"per_base": per_base, "union": union}


def params_for_class(cls: Type[Any], *, skip_composed: bool = True) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Return (required_param_names, optional_param_names, param_annotations_by_name) for a class
    by aggregating across its mixins in MRO order. If skip_composed=True, skip params declared
    on cls itself (e.g., 'SystemElement') to avoid duplicates with 'Identifiable'.
    """
    info = describe_inits_by_mro(cls)

    ann_map: Dict[str, Any] = {}
    required: List[str] = []
    optional: List[str] = []

    # Ordered bases in MRO
    mro_bases = [b for b in cls.__mro__ if b is not object]
    if skip_composed:
        mro_bases = [b for b in mro_bases if b.__name__ != cls.__name__]

    for base in mro_bases:
        rows = info["per_base"].get(base.__name__, [])
        for r in rows:
            pname = r["name"]
            if pname not in ann_map and r.get("annotation", None) is not None:
                ann_map[pname] = r["annotation"]
            if r["required"]:
                if pname not in required:
                    required.append(pname)
            else:
                if pname not in optional:
                    optional.append(pname)

    return required, optional, ann_map


def inits_to_dataframe(cls, include_types: bool = False, include_kind: bool = False) -> pd.DataFrame:
    info = describe_inits_by_mro(cls)

    records = []
    skip_class_name = cls.__name__  # e.g., "SystemElement"

    for mixin, rows in info["per_base"].items():
        if mixin == skip_class_name:
            # Skip entries declared in the composed class
            continue

        for r in rows:
            rec = {
                "mixin": mixin,
                "inputname": r["name"],
                "required": r["required"],
                "default": r["default"],
            }
            if include_types:
                ann = r.get("annotation", None)
                rec["type"] = getattr(ann, "__name__", str(ann)) if ann is not None else None
            if include_kind:
                rec["kind"] = r.get("kind", None)
            records.append(rec)

    df = pd.DataFrame.from_records(records)

    # Sort by MRO order (excluding the composed class), then by parameter name
    mro_order = [b.__name__ for b in cls.__mro__ if b is not object and b.__name__ != skip_class_name]
    order_map = {name: i for i, name in enumerate(mro_order)}
    df["__mixin_order"] = df["mixin"].map(order_map)
    df.sort_values(["__mixin_order", "inputname"], inplace=True, kind="stable")
    df.drop(columns="__mixin_order", inplace=True)

    return df


def create_object(name, classes):
    for perm in itertools.permutations(classes):
        try:
            object = type(name, perm, {})
            return object
        except TypeError:
            pass
    raise TypeError("No valid base class order exists")


def create_vessel(
    env,
    vessel_obj,
    origin = None,
    destination = None,
    route = None,
    show_unused_optional_parameters=False,
    **kwargs,
):
    # Check vessel class
    check_class_is_vesselclass(vessel_obj)

    # Automatically generated constructor arguments take precedence
    auto_kwargs = {
        "env": env,
        "origin": origin,
        "destination": destination,
        "route": route,
    }
    candidate_kwargs = {**kwargs, **auto_kwargs}

    # Inspect constructor parameters
    df, missing_parameters = inspect_object(
        vessel_obj,
        candidate_kwargs,
        False,
    )

    if show_unused_optional_parameters or missing_parameters:
        display(df)

    # Separate constructor arguments from extra metadata
    allowed = get_all_init_params(vessel_obj)

    filtered_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in allowed
    }
    extra_kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key not in allowed
    }

    # Construct vessel
    vessel = vessel_obj(**filtered_kwargs)

    # Store unused parameters as metadata when supported
    if "ExtraMetadata" in {
        cls.__name__ for cls in type(vessel).mro()
    }:
        vessel.metadata.update(extra_kwargs)

    if show_unused_optional_parameters:
        return vessel, df

    return vessel