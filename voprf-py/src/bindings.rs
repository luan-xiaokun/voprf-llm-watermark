use curve25519_dalek::RistrettoPoint;
use group::GroupEncoding;
use pyo3::prelude::*;
use rayon::prelude::*;
use voprf::{
    BlindedElement, EvaluationElement, Proof, Ristretto255, VoprfClient, VoprfClientBlindResult,
    VoprfServer, VoprfServerBatchEvaluateFinishResult,
};

const SERVER_INFO: &[u8] = b"VOPRF-Ristretto255-SHA512-Watermark";

type CipherSuite = Ristretto255;

#[pyclass]
#[pyo3(name = "BlindedElement")]
#[derive(Clone)]
pub struct PyBlindedElement {
    pub message: BlindedElement<CipherSuite>,
}

#[pymethods]
impl PyBlindedElement {
    #[staticmethod]
    fn from_binary(data: &[u8]) -> PyResult<Self> {
        let message = BlindedElement::<CipherSuite>::deserialize(data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "BlindedElement deserialization error: {}",
                e
            ))
        })?;
        Ok(Self { message })
    }

    fn to_binary(&self) -> PyResult<Vec<u8>> {
        Ok(self.message.serialize().to_vec())
    }
}

#[pyclass]
#[pyo3(name = "EvaluationElement")]
#[derive(Clone)]
pub struct PyEvaluationElement {
    pub message: EvaluationElement<CipherSuite>,
}

#[pymethods]
impl PyEvaluationElement {
    #[staticmethod]
    fn from_binary(data: &[u8]) -> PyResult<Self> {
        let message = EvaluationElement::<CipherSuite>::deserialize(data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "EvaluationElement deserialization error: {}",
                e
            ))
        })?;
        Ok(Self { message })
    }

    fn to_binary(&self) -> PyResult<Vec<u8>> {
        Ok(self.message.serialize().to_vec())
    }
}

#[pyclass]
#[pyo3(name = "Proof")]
#[derive(Clone)]
pub struct PyProof {
    pub proof: Proof<CipherSuite>,
}

#[pymethods]
impl PyProof {
    #[staticmethod]
    fn from_binary(data: &[u8]) -> PyResult<Self> {
        let proof = Proof::<CipherSuite>::deserialize(data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Proof deserialization error: {}",
                e
            ))
        })?;
        Ok(Self { proof })
    }

    fn to_binary(&self) -> PyResult<Vec<u8>> {
        Ok(self.proof.serialize().to_vec())
    }
}

#[pyclass]
#[pyo3(name = "PublicKey")]
#[derive(Clone)]
pub struct PyPublicKey {
    pub key: RistrettoPoint,
}

#[pymethods]
impl PyPublicKey {
    #[staticmethod]
    fn from_binary(data: &[u8]) -> PyResult<Self> {
        let key = RistrettoPoint::from_bytes(data.try_into()?)
            .into_option()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid RistrettoPoint bytes")
            })?;
        Ok(Self { key })
    }

    fn to_binary(&self) -> PyResult<Vec<u8>> {
        Ok(self.key.to_bytes().to_vec())
    }
}

#[pyclass]
#[pyo3(name = "VoprfClient")]
#[derive(Clone)]
pub struct PyVoprfClient {
    pub state: VoprfClient<CipherSuite>,
}

#[pymethods]
impl PyVoprfClient {
    #[staticmethod]
    fn from_binary(data: &[u8]) -> PyResult<Self> {
        let state = VoprfClient::<CipherSuite>::deserialize(data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "VoprfClient deserialization error: {}",
                e
            ))
        })?;
        Ok(Self { state })
    }

    fn to_binary(&self) -> PyResult<Vec<u8>> {
        Ok(self.state.serialize().to_vec())
    }
}

#[pyclass]
#[pyo3(name = "VoprfServer")]
#[derive(Clone)]
pub struct PyVoprfServer {
    pub server: VoprfServer<CipherSuite>,
}

#[pymethods]
impl PyVoprfServer {
    #[new]
    fn new(server_seed: &[u8]) -> PyResult<Self> {
        let server = VoprfServer::<CipherSuite>::new_from_seed(
            server_seed,
            b"VOPRF-Ristretto255-SHA512-Watermark",
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Server initialization error: {}",
                e
            ))
        })?;
        Ok(Self { server })
    }

    #[staticmethod]
    fn derive_key(server_seed: &[u8]) -> PyResult<Vec<u8>> {
        let key = voprf::derive_key::<CipherSuite>(server_seed, SERVER_INFO, voprf::Mode::Voprf)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Key derivation error: {}",
                    e
                ))
            })?;
        Ok(key.as_bytes().to_vec())
    }

    fn get_public_key(&self) -> PyPublicKey {
        PyPublicKey {
            key: self.server.get_public_key(),
        }
    }

    fn batch_evaluate(&self, inputs: Vec<Vec<u8>>) -> PyResult<Vec<Vec<u8>>> {
        let results: Result<Vec<Vec<u8>>, voprf::Error> = inputs
            .par_iter()
            .map(|input| self.server.evaluate(input).map(|output| output.to_vec()))
            .collect();
        results.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Batch evaluation error: {}",
                e
            ))
        })
    }

    fn evaluate(&self, input: &[u8]) -> PyResult<Vec<u8>> {
        self.server
            .evaluate(input)
            .map(|output| output.to_vec())
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Evaluation error: {}", e))
            })
    }

    fn batch_blind_evaluate(
        &self,
        messages: Vec<PyBlindedElement>,
    ) -> PyResult<(Vec<PyEvaluationElement>, PyProof)> {
        let mut server_rng = &mut rand::thread_rng();
        let prepared_elements: Vec<_> = self
            .server
            .batch_blind_evaluate_prepare(messages.iter().map(|msg| &msg.message))
            .collect();
        let VoprfServerBatchEvaluateFinishResult { messages, proof } = self
            .server
            .batch_blind_evaluate_finish(
                &mut server_rng,
                messages.iter().map(|msg| &msg.message),
                &prepared_elements,
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Batch blind evaluation error: {}",
                    e
                ))
            })?;
        let evaluation_results = messages
            .into_iter()
            .map(|msg| PyEvaluationElement {
                message: msg.clone(),
            })
            .collect();
        Ok((evaluation_results, PyProof { proof }))
    }
}

#[pyfunction]
pub fn prepare_batch_blind_inputs(
    inputs: Vec<Vec<u8>>,
) -> PyResult<(Vec<PyVoprfClient>, Vec<PyBlindedElement>)> {
    let mut rng = rand::thread_rng();
    let blind_results: Vec<VoprfClientBlindResult<CipherSuite>> = inputs
        .iter()
        .map(|input| VoprfClient::<CipherSuite>::blind(input, &mut rng))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Batch blind input preparation error: {}",
                e
            ))
        })?;
    let clients: Vec<PyVoprfClient> = blind_results
        .iter()
        .map(|result| PyVoprfClient {
            state: result.state.clone(),
        })
        .collect();
    let blinded_elements: Vec<PyBlindedElement> = blind_results
        .iter()
        .map(|result| PyBlindedElement {
            message: result.message.clone(),
        })
        .collect();
    Ok((clients, blinded_elements))
}

#[pyfunction]
pub fn finalize_batch_blind_results(
    inputs: Vec<Vec<u8>>,
    states: Vec<PyVoprfClient>,
    messages: Vec<PyEvaluationElement>,
    proof: PyProof,
    public_key: PyPublicKey,
) -> PyResult<Vec<Vec<u8>>> {
    let states = states
        .into_iter()
        .map(|state| state.state)
        .collect::<Vec<_>>();
    let messages = messages
        .into_iter()
        .map(|msg| msg.message)
        .collect::<Vec<_>>();
    let proof = proof.proof;
    let public_key = public_key.key;
    let client_batch_finalize_results: Result<Vec<_>, _> =
        VoprfClient::<CipherSuite>::batch_finalize(&inputs, &states, &messages, &proof, public_key)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Batch blind result finalization error: {}",
                    e
                ))
            })?
            .collect();
    let res = match client_batch_finalize_results {
        Ok(results) => results.into_iter().map(|r| r.to_vec()).collect(),
        Err(e) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Batch blind result finalization error: {}",
                e
            )));
        }
    };
    Ok(res)
}
