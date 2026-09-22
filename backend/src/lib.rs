use pyo3::prelude::*;

#[pyfunction]
fn acceleration(a: usize, b: usize) -> PyResult<usize> {
    // Matrix multiplications
    Ok(a + b)
}

#[pymodule]
fn backend(_py: Python, m: &PyModule) -> PyResult<()> {
    // add function on the module
    m.add_function(wrap_pyfunction!(acceleration, m)?)?;
    Ok(())
}