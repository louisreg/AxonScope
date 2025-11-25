# Changelog

All notable changes to this project are documented here.

## [Unreleased]

**Note:** Changes since the last release.

* ...

---

## [0.2.0] - 2025-11-25

**Note:** This is the first version whose changes are formally tracked in this file.

### Added (New Features)

* **Custom Ion Channels:** Introduced the `IonChannelModelBase` and `CompositeICM` classes to allow users to **create custom ion channel models** and **composite models**.
* **Sundt Model:** Initial implementation of the **Sundt model** as seen in NRV (Note: There is a currently a known issue with the model behavior that requires further debugging).

### Changed (Improvements/Updates)

* **Core Integration:** The `Axon` and `Solver` classes have been updated to accept and utilize the new custom ion channel model classes.
* **Performance Optimization:** The **Solver has been modified to utilize JAX 100** with the implementation of a dedicated **tridiagonal solver**. This combination yields the best performance results (see the `./benchmark/CrankNicholson_runtime/` directory).

### Fixed (Bug Fixes)

* *N/A*

---

## [0.1.0] and Earlier (Historical)

**Note:** Changes made during initial development (including base model validation and initial solver implementation) were not formally documented in this file. Please refer to the commit history for details on these earlier developments.