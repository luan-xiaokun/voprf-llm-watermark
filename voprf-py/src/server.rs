use curve25519_dalek::RistrettoPoint;
use rayon::prelude::*;
use voprf::{
    Ristretto255, VoprfClient, VoprfClientBlindResult, VoprfServer,
    VoprfServerBatchEvaluateFinishResult, VoprfServerBatchEvaluateResult, derive_key,
};

type CipherSuite = Ristretto255;

const SERVER_INFO: &[u8] = b"VOPRF-Ristretto255-SHA512-Watermark";

pub struct Server {
    voprf_server: VoprfServer<CipherSuite>,
}

impl Server {
    pub fn new(server_seed: &[u8; 32]) -> Result<Self, voprf::Error> {
        let voprf_server = VoprfServer::<CipherSuite>::new_from_seed(server_seed, SERVER_INFO)?;
        Ok(Self { voprf_server })
    }

    pub fn derive_key(server_seed: &[u8; 32]) -> Result<Vec<u8>, voprf::Error> {
        let key = derive_key::<CipherSuite>(server_seed, SERVER_INFO, voprf::Mode::Voprf)?;
        Ok(key.as_bytes().to_vec())
    }

    pub fn get_public_key(&self) -> RistrettoPoint {
        self.voprf_server.get_public_key()
    }

    pub fn batch_evaluate(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, voprf::Error> {
        let results: Result<Vec<Vec<u8>>, voprf::Error> = inputs
            .par_iter()
            .map(|&input| {
                self.voprf_server
                    .evaluate(input)
                    .map(|output| output.to_vec())
            })
            .collect();
        results
    }

    pub fn batch_blind_evaluate(
        &self,
        client_blind_results: &[VoprfClientBlindResult<CipherSuite>],
    ) -> Result<VoprfServerBatchEvaluateResult<CipherSuite>, voprf::Error> {
        let mut server_rng = &mut rand::thread_rng();
        let messages: Vec<_> = client_blind_results
            .iter()
            .map(|result| result.message.clone())
            .collect();
        let prepared_evaluation_elements = self
            .voprf_server
            .batch_blind_evaluate_prepare(messages.iter());
        let prepared_elements: Vec<_> = prepared_evaluation_elements.collect();
        let VoprfServerBatchEvaluateFinishResult { messages, proof } = self
            .voprf_server
            .batch_blind_evaluate_finish(&mut server_rng, messages.iter(), &prepared_elements)
            .expect("Unable to perform server batch evaluate");
        let messages: Vec<_> = messages.collect();
        Ok(VoprfServerBatchEvaluateResult { messages, proof })
    }
}

pub fn prepare_batch_blind_inputs(
    inputs: &[&[u8]],
) -> Result<Vec<VoprfClientBlindResult<CipherSuite>>, voprf::Error> {
    let mut rng = rand::thread_rng();
    inputs
        .iter()
        .map(|&input| VoprfClient::<CipherSuite>::blind(input, &mut rng))
        .collect()
}

pub fn finalize_batch_blind_results(
    inputs: &[&[u8]],
    client_blind_results: &[VoprfClientBlindResult<CipherSuite>],
    server_batch_evaluation_results: &VoprfServerBatchEvaluateResult<CipherSuite>,
    server: &Server,
) -> Result<Vec<Vec<u8>>, voprf::Error> {
    let input_length = inputs.len();
    let inputs: Vec<&[u8]> = inputs.iter().take(input_length).cloned().collect();
    let clients: Vec<VoprfClient<CipherSuite>> = client_blind_results
        .iter()
        .map(|result| result.state.clone())
        .collect();
    let messages = &server_batch_evaluation_results.messages;
    let proof = &server_batch_evaluation_results.proof;
    let client_batch_finalize_results: Result<Vec<_>, _> =
        VoprfClient::<CipherSuite>::batch_finalize(
            &inputs,
            &clients,
            messages,
            proof,
            server.get_public_key(),
        )?
        .collect();
    let res = match client_batch_finalize_results {
        Ok(owned_data) => {
            let slices: Vec<Vec<u8>> = owned_data.iter().map(|ga| ga.to_vec()).collect();
            slices
        }
        Err(e) => return Err(e),
    };
    Ok(res)
}
