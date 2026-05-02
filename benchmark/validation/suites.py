from benchmark.nrv_performance.suites import (
    NRV_PERFORMANCE_SUITES,
    SUITE_ALIASES,
    NrvPerformanceSuite,
    resolve_suite,
    suite_choices,
)

ValidationSuite = NrvPerformanceSuite
VALIDATION_SUITES = {
    **NRV_PERFORMANCE_SUITES,
    **{alias: resolve_suite(alias) for alias in SUITE_ALIASES},
}

__all__ = [
    "NRV_PERFORMANCE_SUITES",
    "SUITE_ALIASES",
    "NrvPerformanceSuite",
    "ValidationSuite",
    "VALIDATION_SUITES",
    "resolve_suite",
    "suite_choices",
]
