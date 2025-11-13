use crank_nicholson_rust::params;
use std::time::Instant;
use crank_nicholson_rust::sim::{hines_basic, hines_tridiag, hines_tridiag_fast, hines_tridiag_rayon};
use crank_nicholson_rust::plot::{plot_results};


// --------------------------
// Main
// --------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nx_v = vec![11,21,51,101,201,501,1001];

    for &nx in &nx_v {
        let start = Instant::now();
        let (t_vec, V_all) = hines_tridiag_fast(nx, &params::SimParams::new(nx));
        let duration = start.elapsed();
        println!("Nx = {}, simulation time = {:.3} s", nx, duration.as_secs_f32());

        if nx == 101 {
            plot_results(&t_vec, &V_all, &[nx / 4, nx / 2, 3 * nx / 4])?;
        }
    }

    Ok(())
}
