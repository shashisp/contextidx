use pyo3::prelude::*;

/// Batch-compute decay scores for N units.
///
/// Each parallel vector must have the same length.
/// `model` is one of "exponential", "linear", or "step".
/// For step decay, the score is simply confidence if age_days < half_life, else 0.
#[pyfunction]
fn batch_decay(
    confidences: Vec<f64>,
    decay_rates: Vec<f64>,
    age_days: Vec<f64>,
    reinforcement_counts: Vec<i32>,
    model: &str,
    reinforcement_factor: f64,
) -> PyResult<Vec<f64>> {
    let n = confidences.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let effective_age = if reinforcement_counts[i] > 0 {
            age_days[i] * reinforcement_factor.powi(reinforcement_counts[i])
        } else {
            age_days[i]
        };

        let score = match model {
            "exponential" => confidences[i] * (-decay_rates[i] * effective_age).exp(),
            "linear" => {
                let half_life = if decay_rates[i] > 0.0 {
                    (2.0_f64).ln() / decay_rates[i]
                } else {
                    f64::INFINITY
                };
                confidences[i] * (1.0 - effective_age / half_life).max(0.0)
            }
            "step" => {
                let half_life = if decay_rates[i] > 0.0 {
                    (2.0_f64).ln() / decay_rates[i]
                } else {
                    f64::INFINITY
                };
                if effective_age < half_life {
                    confidences[i]
                } else {
                    0.0
                }
            }
            _ => confidences[i],
        };
        out.push(score);
    }
    Ok(out)
}

/// Batch-compute composite retrieval scores for N candidates.
///
/// `weights` order: [semantic, bm25, recency, confidence, decay, reinforcement]
#[pyfunction]
fn batch_score(
    semantic_scores: Vec<f64>,
    recency_scores: Vec<f64>,
    confidences: Vec<f64>,
    decay_scores: Vec<f64>,
    reinforcement_scores: Vec<f64>,
    bm25_scores: Vec<f64>,
    weights: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let n = semantic_scores.len();
    let mut out = Vec::with_capacity(n);

    let w_sem = weights[0];
    let w_bm25 = weights[1];
    let w_rec = weights[2];
    let w_conf = weights[3];
    let w_dec = weights[4];
    let w_reinf = weights[5];

    for i in 0..n {
        let raw = w_sem * semantic_scores[i]
            + w_bm25 * bm25_scores[i]
            + w_rec * recency_scores[i]
            + w_conf * confidences[i]
            + w_dec * decay_scores[i]
            + w_reinf * reinforcement_scores[i];
        out.push(raw.clamp(0.0, 1.0));
    }
    Ok(out)
}

/// Batch cosine similarity: one query vector against N candidate vectors.
///
/// `candidates_flat` is the row-major flattened matrix of N vectors, each of
/// length `dim`.  Returns N similarity values.  Zero-norm vectors yield 0.0.
#[pyfunction]
fn batch_cosine_similarity(
    query: Vec<f64>,
    candidates_flat: Vec<f64>,
    dim: usize,
) -> PyResult<Vec<f64>> {
    if dim == 0 {
        return Ok(vec![]);
    }
    let n = candidates_flat.len() / dim;
    let mut out = Vec::with_capacity(n);

    let mut q_norm_sq: f64 = 0.0;
    for &v in &query {
        q_norm_sq += v * v;
    }
    let q_norm = q_norm_sq.sqrt();
    if q_norm == 0.0 {
        return Ok(vec![0.0; n]);
    }

    for row in 0..n {
        let offset = row * dim;
        let mut dot: f64 = 0.0;
        let mut c_norm_sq: f64 = 0.0;
        for j in 0..dim {
            let cj = candidates_flat[offset + j];
            dot += query[j] * cj;
            c_norm_sq += cj * cj;
        }
        let c_norm = c_norm_sq.sqrt();
        if c_norm == 0.0 {
            out.push(0.0);
        } else {
            out.push(dot / (q_norm * c_norm));
        }
    }
    Ok(out)
}

/// Batch rule-based contradiction detection.
///
/// Returns a Vec<bool> where true means a contradiction was detected
/// between `new_content` and that existing content string.
#[pyfunction]
fn detect_contradictions(new_content: &str, existing_contents: Vec<String>) -> PyResult<Vec<bool>> {
    let new_lower = new_content.to_lowercase();
    let new_words: std::collections::HashSet<&str> = new_lower.split_whitespace().collect();
    let new_has_neg = has_negation(&new_lower);

    let mut out = Vec::with_capacity(existing_contents.len());

    for existing in &existing_contents {
        let ex_lower = existing.to_lowercase();

        // Check verb-pair contradictions
        if check_verb_pairs(&new_lower, &ex_lower) {
            out.push(true);
            continue;
        }

        // Check high overlap + opposite negation
        let ex_words: std::collections::HashSet<&str> = ex_lower.split_whitespace().collect();
        let overlap = new_words.intersection(&ex_words).count();
        let max_len = new_words.len().max(ex_words.len());

        if max_len > 0 {
            let overlap_ratio = overlap as f64 / max_len as f64;
            let ex_has_neg = has_negation(&ex_lower);
            if overlap_ratio > 0.5 && new_has_neg != ex_has_neg {
                out.push(true);
                continue;
            }
        }

        out.push(false);
    }
    Ok(out)
}

fn has_negation(s: &str) -> bool {
    for word in s.split_whitespace() {
        match word {
            "not" | "no" | "never" | "don't" | "dont" => return true,
            _ => {}
        }
    }
    false
}

fn check_verb_pairs(a: &str, b: &str) -> bool {
    const PAIRS: &[(&str, &str)] = &[
        ("prefers ", "does not prefer "),
        ("likes ", "does not like "),
        ("wants ", "does not want "),
        ("is a ", "is no longer a "),
    ];

    for (pos, neg) in PAIRS {
        let a_pos = a.contains(pos);
        let b_neg = b.contains(neg);
        if a_pos && b_neg {
            return true;
        }
        let a_neg = a.contains(neg);
        let b_pos = b.contains(pos);
        if a_neg && b_pos {
            return true;
        }
    }
    false
}

#[pymodule]
fn _core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_decay, m)?)?;
    m.add_function(wrap_pyfunction!(batch_score, m)?)?;
    m.add_function(wrap_pyfunction!(batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(detect_contradictions, m)?)?;
    Ok(())
}
