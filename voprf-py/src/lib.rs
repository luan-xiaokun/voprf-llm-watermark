use pyo3::prelude::*;

mod bindings;
use bindings::{
    PyBlindedElement, PyEvaluationElement, PyProof, PyPublicKey, PyVoprfClient, PyVoprfServer,
    finalize_batch_blind_results, prepare_batch_blind_inputs,
};

#[pymodule]
#[pyo3(name = "_voprf_py")]
fn voprf_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBlindedElement>()?;
    m.add_class::<PyEvaluationElement>()?;
    m.add_class::<PyProof>()?;
    m.add_class::<PyPublicKey>()?;
    m.add_class::<PyVoprfClient>()?;
    m.add_class::<PyVoprfServer>()?;

    m.add_function(wrap_pyfunction!(prepare_batch_blind_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_batch_blind_results, m)?)?;

    Ok(())
}
