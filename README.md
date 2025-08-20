# AxonScope

**AxonScope** is an **on-going project** (currently in its early development stage) that aims to provide a **modern, user-friendly, and efficient Python module** for the simulation of realistic axons, based on well-established models from the literature.  

---

## Goals

- Provide a **dedicated framework** for axon modeling (unlike general-purpose simulators such as NEURON).  
- Offer a **simple and modern Python interface** with **high performance**.  
- Leverage **modern computing architectures**:
  - **GPUs**  
  - **Massive CPU multithreading**  
  - **Hardware accelerators** (e.g., TPUs).  

---

## Features (planned)

- **Single-compartment and multi-compartment axon models**.  
- **Myelinated and unmyelinated fibers**.  
- **Intracellular and extracellular stimulation mechanisms**.  
- Performance comparable to NEURON, but **simpler and more modern**.  

---

## TODO Roadmap

- [x] Validate base implementation with a passive membrane
- [x] Validate base implementation of Hodgkin-Huxley and Rattay-Aberham models
  - [x]  Check AP shape
  - [x]  Check propagation velocity against NRV  
    - We observe about 10% difference but we use difference velocity estimation method and different solver (for now).
- [x] Add performance benchmarking tooling  
      - Basic decorator for benchmarking is implemented, based on pyinstrument with filtering capabilities. See [this example](./benchmark/simple_benchmark.py). More features will be added to fully utilize pyinstrument capabilities. We might also implement memray for memory tracing.

- [ ] Replace Explicit Euler solver with Crank–Nicholson + Thomas' algorithm  
- [ ] Implement extracellular stimulation mechanisms  
- [ ] Start integrating multicompartment models  
- [ ] Benchmark different backends:  
  - [ ] NumPy  
  - [ ] PyTorch/Pytorch compile
  - [ ] JAX  
  - [ ] Rust  
- [ ] And many more things to do...