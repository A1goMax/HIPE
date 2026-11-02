# HIPE Claims and Required Evidence

MS01 is the supplied 19-page conference manuscript. Page numbers below refer to PDF pages. Its claims are targets for review and testing, not conclusions newly established by this project.

| Claim | Reference | Evidence to recover | Journal investigation | Important boundary |
| --- | --- | --- | --- | --- |
| Gaussian place fields imply frequency-dependent grid-to-place weights | MS01 section 2, Figure 2, pages 2–3 | Derivation and numerical reconstruction with stated domain and basis | Finite frequency sets, truncation and boundary tests | Infinite-domain approximation does not automatically describe finite biological modules |
| Broad fields generalize better with few samples while narrow fields favor precision | Section 3.1, Figure 4, page 3 | Complete task generator, readout, train/test samples and curves | Sample-size sweeps under matched task conditions | Benefit depends on task spectrum and representation normalization |
| Optimal field scale grows with task scale | Section 3.2–3.3, Figures 5–6, equation 13, pages 4–5 | Theory assumptions and simulation grids; Figure 5 reports 1,000 repeats and SEM | Compare predicted and measured optimum across task families | A dominant-mode approximation is not a general theorem for arbitrary tasks |
| Heterogeneous scales improve performance across environments | Section 3.4, Figure 7, page 5 | Scale distributions, five tasks, 1,000 repeats and statistical analysis | Matched resource and total-feature-count comparisons | Best average performance is different from uniform dominance on every task |
| HIPE defines a broader positional kernel | Section 4, equations 17–19, pages 5–6 | Exact scaling, frequency grid and RMS normalization used in code | Discrete-kernel analysis and quantitative width/energy measures | The isotropic-energy positional model is an assumption; trained attention need not satisfy it |
| Standard RoPE is recovered as sigma tends to zero | Section 4 and Appendix C, pages 5–6 and 16–18 | Interpret the formula and implementation consistently | Resolve the finite-grid limit explicitly before a journal claim | Equation 19 retains the square-root frequency factor when sigma is zero; a single normalization constant cannot make unequal frequency weights all equal |
| HIPE helps associative and block retrieval | Table 1, page 7 | Four method/task cells, checkpoint and task distribution | Distance-controlled tests and held-out task settings | A visualization alone does not show a causal retrieval mechanism |
| HIPE lowers LM perplexity under a restricted data budget | Table 2, page 8 | Exact preprocessing, budgets, methods, sigma selection and evaluation | Budget curves, paired seeds, fair tuning and threshold analysis | A lower PPL at one budget does not quantify tokens saved |
| The discrete sharp-to-broad hierarchy helps | Table 3, page 8 | Exact eight-layer schedules and common training/evaluation | Equal-energy controls and order interventions | Schedule comparison must keep search effort and other settings matched |
| HIPE changes deep-layer attention organization | Appendix D, Figure 9, page 18 | Same-input checkpoint comparisons and plotting code | Attention-distance summaries and task interventions across seeds | Diffuse attention can accompany a result without causing it |

## Theory issue requiring a recorded decision

MS01 equation 19 gives a scale proportional to `sqrt(omega * exp(-sigma^2 * omega^2 / 2))`. At sigma zero, the exponential becomes one while the `sqrt(omega)` factor remains. On the finite log-spaced grid used by RoPE, this does not directly equal the uniform scale of standard RoPE. Appendix C also treats frequency-domain limits that need to be checked against the actual sampled frequencies and bandwidth.

The journal work should distinguish the unweighted RoPE reference, the density-compensated spectral construction, and any broader family that contains both. The correct resolution may be a narrower mathematical statement, a revised family definition, or a separately named control. Historical code and reported results must keep their original semantics while this is investigated.

## Evidence labels

Use `reported` for a manuscript number, `recovered` for an identified historical artifact, `replicated` for a newly executed comparison, and `confirmed` only after the relevant scientific and provenance checks. Historical code associations are separately labeled `verified`, `inferred`, or `unknown`.

A log containing a commit string is not by itself proof that the training code came from that commit. It may describe later evaluation, or a dirty checkout. Evidence links must state what the record identifies.
