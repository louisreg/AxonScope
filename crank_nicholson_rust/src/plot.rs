use ndarray::{Array1, Array2};
use plotters::prelude::*;

pub fn plot_results(
    t_vec: &Array1<f64>,
    V_all: &Array2<f64>,
    positions: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("axon_voltage.png", (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Axon Voltage over Time", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(t_vec[0]..t_vec[t_vec.len() - 1], -90.0f64..90.0f64)?;

    chart.configure_mesh().x_desc("Time [ms]").y_desc("Voltage [mV]").draw()?;

    let color_list = vec![RED, BLUE, GREEN, MAGENTA, CYAN, BLACK];
    for (i, &pos_idx) in positions.iter().enumerate() {
        let series_color = color_list[i % color_list.len()];
        let series: Vec<(f64, f64)> = t_vec
            .iter()
            .zip(V_all.column(pos_idx).iter())
            .map(|(&t, &v)| (t, v))
            .collect();
        chart.draw_series(LineSeries::new(series, series_color))?;
    }

    Ok(())
}