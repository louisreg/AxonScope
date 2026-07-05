"""Experimental pseudo-double-cable validation helpers."""

from benchmark.pseudo_double.plotting import write_validation_plots
from benchmark.pseudo_double.reductions import (
    DoubleCableBlockCoefficients,
    SchurLocalReduction,
    double_cable_coefficients_from_solver_terms,
    schur_local_v1,
    series_equivalent,
    tridiagonal_edges_to_jax,
)
from benchmark.pseudo_double.schur_runner import (
    PseudoDoubleSchurLocalConfig,
    run_schur_local_population,
)
from benchmark.pseudo_double.single_chain import (
    PseudoDoubleSegmentType,
    PseudoDoubleSingleChainConfig,
    build_pseudo_double_single_chain_mrg,
    segment_scaled_point_source_stimulation,
    single_chain_segment_counts,
    single_chain_segment_type,
    single_chain_vext_alpha,
)
from benchmark.pseudo_double.series_runner import (
    PseudoDoubleSeriesConfig,
    run_series_population,
)
from benchmark.pseudo_double.validation import (
    IMPLEMENTED_VALIDATION_MODES,
    PSEUDO_DOUBLE_EXPERIMENTAL_MODES,
    PseudoDoubleEffectiveConfig,
    PseudoDoubleSplitConfig,
    VALIDATION_MODES,
    build_validation_population,
    calibrate_pseudo_double_effective,
    calibrate_pseudo_double_schur_local,
    calibrate_pseudo_double_series,
    calibrate_pseudo_double_single_chain,
    calibrate_pseudo_double_split,
    compare_mode_results,
    mode_metadata,
    normalize_validation_mode,
    run_validation,
    score_validation_result,
)

__all__ = [
    "DoubleCableBlockCoefficients",
    "IMPLEMENTED_VALIDATION_MODES",
    "PSEUDO_DOUBLE_EXPERIMENTAL_MODES",
    "PseudoDoubleSegmentType",
    "PseudoDoubleEffectiveConfig",
    "PseudoDoubleSchurLocalConfig",
    "PseudoDoubleSingleChainConfig",
    "PseudoDoubleSeriesConfig",
    "PseudoDoubleSplitConfig",
    "SchurLocalReduction",
    "VALIDATION_MODES",
    "build_pseudo_double_single_chain_mrg",
    "build_validation_population",
    "calibrate_pseudo_double_effective",
    "calibrate_pseudo_double_schur_local",
    "calibrate_pseudo_double_series",
    "calibrate_pseudo_double_single_chain",
    "calibrate_pseudo_double_split",
    "compare_mode_results",
    "double_cable_coefficients_from_solver_terms",
    "mode_metadata",
    "normalize_validation_mode",
    "run_validation",
    "run_schur_local_population",
    "run_series_population",
    "schur_local_v1",
    "score_validation_result",
    "segment_scaled_point_source_stimulation",
    "series_equivalent",
    "single_chain_segment_counts",
    "single_chain_segment_type",
    "single_chain_vext_alpha",
    "tridiagonal_edges_to_jax",
    "write_validation_plots",
]
