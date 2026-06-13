//! Pure-Rust inference for the trained PolicyValueNet (CTNN format,
//! exported by training/export_net.py). The net is tiny (~1–2M MACs), so a
//! straightforward autovectorized matmul is plenty — no framework needed.
//!
//! The file embeds a self-check vector; `load` recomputes it and panics on
//! mismatch, so stale or corrupted weights can never silently play games.

use crate::codec::NUM_ACTIONS;
use crate::obs::OBS_DIM;

pub struct MlpNet {
    pub hidden: usize,
    w1t: Vec<f32>, // [OBS_DIM][hidden] - input-major (transposed at load)
    b1: Vec<f32>,
    w2t: Vec<f32>, // [hidden][hidden] - input-major (transposed at load)
    b2: Vec<f32>,
    wp: Vec<f32>, // [NUM_ACTIONS][hidden]
    bp: Vec<f32>,
    wv: Vec<f32>, // [hidden]
    bv: f32,
}

/// Reusable forward-pass scratch (one per bot; keeps play allocation-free).
#[derive(Clone)]
pub struct NetScratch {
    h1: Vec<f32>,
    h2: Vec<f32>,
}

impl NetScratch {
    pub fn new(net: &MlpNet) -> NetScratch {
        NetScratch {
            h1: vec![0.0; net.hidden],
            h2: vec![0.0; net.hidden],
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // A strict left-to-right f32 sum cannot be vectorized because float
    // addition is not associative. Independent lanes let LLVM vectorize.
    let mut acc = [0.0f32; 16];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (xs, ys) in ca.by_ref().zip(cb.by_ref()) {
        for lane in 0..16 {
            acc[lane] += xs[lane] * ys[lane];
        }
    }
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    acc.iter().sum::<f32>() + tail
}

fn axpy(out: &mut [f32], input: f32, column: &[f32]) {
    for (value, &weight) in out.iter_mut().zip(column) {
        *value += input * weight;
    }
}

fn matvec_t_relu(weights: &[f32], input: &[f32], bias: &[f32], out: &mut [f32]) {
    let outputs = out.len();
    debug_assert_eq!(weights.len(), input.len() * outputs);
    out.copy_from_slice(bias);
    for (column, &input_value) in input.iter().enumerate() {
        if input_value != 0.0 {
            axpy(
                out,
                input_value,
                &weights[column * outputs..(column + 1) * outputs],
            );
        }
    }
    for value in out {
        *value = value.max(0.0);
    }
}

fn transpose(weights: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; weights.len()];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = weights[row * columns + column];
        }
    }
    out
}

fn read_f32s(bytes: &[u8], off: &mut usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = *off + i * 4;
        out.push(f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()));
    }
    *off += n * 4;
    out
}

impl MlpNet {
    pub fn load(path: &std::path::Path) -> MlpNet {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("cannot read net {}: {e}", path.display()));
        assert_eq!(&bytes[0..4], b"CTNN", "not a CTNN file");
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        assert_eq!(u32_at(4), 1, "CTNN version mismatch");
        assert_eq!(u32_at(8) as usize, OBS_DIM, "obs_dim mismatch");
        assert_eq!(u32_at(12) as usize, NUM_ACTIONS, "num_actions mismatch");
        let hidden = u32_at(16) as usize;

        let mut off = 20;
        let net = MlpNet {
            hidden,
            w1t: transpose(
                &read_f32s(&bytes, &mut off, hidden * OBS_DIM),
                hidden,
                OBS_DIM,
            ),
            b1: read_f32s(&bytes, &mut off, hidden),
            w2t: transpose(
                &read_f32s(&bytes, &mut off, hidden * hidden),
                hidden,
                hidden,
            ),
            b2: read_f32s(&bytes, &mut off, hidden),
            wp: read_f32s(&bytes, &mut off, NUM_ACTIONS * hidden),
            bp: read_f32s(&bytes, &mut off, NUM_ACTIONS),
            wv: read_f32s(&bytes, &mut off, hidden),
            bv: read_f32s(&bytes, &mut off, 1)[0],
        };

        // Embedded self-check: recompute, refuse drifted weights.
        let test_obs = read_f32s(&bytes, &mut off, OBS_DIM);
        let expect_value = read_f32s(&bytes, &mut off, 1)[0];
        let expect_logits = read_f32s(&bytes, &mut off, 8);
        let mut scratch = NetScratch::new(&net);
        net.trunk(&test_obs, &mut scratch);
        // Compare raw (unclamped) — the test vector is an OOD probe.
        let value = dot(&net.wv, &scratch.h2) + net.bv;
        assert!(
            (value - expect_value).abs() < 1e-3,
            "CTNN self-check failed: value {value} != {expect_value}"
        );
        for (i, &e) in expect_logits.iter().enumerate() {
            let logit = dot(&net.wp[i * hidden..(i + 1) * hidden], &scratch.h2) + net.bp[i];
            assert!(
                (logit - e).abs() < 1e-3,
                "CTNN self-check failed: logit {i}"
            );
        }
        net
    }

    /// Run the shared trunk; results live in `scratch` for the heads below.
    pub fn trunk(&self, obs: &[f32], scratch: &mut NetScratch) {
        debug_assert_eq!(obs.len(), OBS_DIM);
        matvec_t_relu(&self.w1t, obs, &self.b1, &mut scratch.h1);
        matvec_t_relu(&self.w2t, &scratch.h1, &self.b2, &mut scratch.h2);
    }

    /// Value head (expected terminal reward for the acting seat, ~[-1, 1]).
    pub fn value_from(&self, scratch: &NetScratch) -> f32 {
        (dot(&self.wv, &scratch.h2) + self.bv).clamp(-1.0, 1.0)
    }

    /// Policy logit for one action id.
    pub fn logit_from(&self, scratch: &NetScratch, action_id: usize) -> f32 {
        dot(
            &self.wp[action_id * self.hidden..(action_id + 1) * self.hidden],
            &scratch.h2,
        ) + self.bp[action_id]
    }
}

#[cfg(test)]
mod tests {
    use super::dot;

    #[test]
    fn optimized_dot_matches_scalar() {
        for len in [1, 15, 16, 17, 192, 299, 512, 1350] {
            let a: Vec<f32> = (0..len)
                .map(|i| ((i * 17 % 31) as f32 - 15.0) / 13.0)
                .collect();
            let b: Vec<f32> = (0..len)
                .map(|i| ((i * 11 % 29) as f32 - 14.0) / 9.0)
                .collect();
            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let actual = dot(&a, &b);
            assert!(
                (actual - expected).abs() <= 1e-3 * expected.abs().max(1.0),
                "len {len}: optimized {actual} != scalar {expected}"
            );
        }
    }
}
