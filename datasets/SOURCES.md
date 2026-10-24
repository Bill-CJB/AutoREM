# Dataset sources and construction

## Generated reformulation data

Random64, Hard32, Large32, and Validation64 use the robust linear-program generator distributed with this repository. The generator samples nominal models and box, budget, and polyhedral uncertainty, computes solver-based reference objectives, and renders the model descriptions. Hard32 stresses polyhedral constraints; the released Large32 contains 16 six-variable and 16 seven-variable instances. Validation64 supports validation-based acceptance during offline memory adaptation.

General-P32 preserves the original uncertain supports while introducing non-diagonal, full-rank loading blocks in existing uncertain units. Deterministic units remain deterministic. Coupled-Row32 shares latent uncertainty across multiple uncertain rows. These are structural extensions of the same generator family.

General-P32 was chosen after inspecting model outputs on a generated candidate set. Coupled-Row reporting also involved output-informed run selection. These choices make both sets exploratory. The released files fix the complete 32-instance inputs; users should evaluate every instance and report all repeated runs.

## Application32

Application32 contains two robust variants of each of 16 nominal problems: eight from IndustryOR and eight from OptiBench. The resulting 32 cases contain 16 IndustryOR and 16 OptiBench records. Nominal problems were curated for interpretable uncertain coefficients, including objective and multi-row uncertainty. One variant follows the numerical uncertainty-generation procedure; the other uses literature-motivated parameter settings. The released descriptions include the full nominal problem and appended robust extension.

- **IndustryOR:** Huang et al., *ORLM: A Customizable Framework in Training Large Models for Automated Optimization Modeling*. Operations Research, 73(6):2986--3009, 2025. The record-level `source` value identifies these cases.
- **OptiBench:** Yang et al., *OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling*. International Conference on Learning Representations, 2025. The record-level `source` value identifies these cases.

The data are a small curated stress test of supported uncertainty families, rather than an exhaustive sample of industrial optimization. The fixed application descriptions and target objectives are distributed directly; exact regeneration of these curated natural-language cases is not claimed.

## ExternalRO32

ExternalRO32 contains 32 numerical robust LP and MILP instances from five publications or author-released benchmark sources. It preserves source numerical models and uncertainty sets in compact representations, with the complete numerical arrays and sparse rows required to specify the problems. It contains 20 LPs and 12 MILPs, with 14 box, eight budget, seven polyhedral, and three box--polyhedral intersection instances.

| Source | Count | Selected structural block |
| --- | ---: | --- |
| Ben-Tal, El Ghaoui, and Nemirovski (2009) | 4 | Drug-production example and three portfolio variants |
| Li, Ding, and Floudas (2011) | 12 | LP/MILP, uncertainty-location, and box/polyhedral structural blocks |
| Li, Tang, and Floudas (2012) | 4 | Multiperiod-production variants |
| Goerigk and Khosravi (2024) | 6 | Three selection samplers crossed with budget endpoints 5 and 20 at official index 1 |
| Ben-Tal and Nemirovski (2000), with NETLIB models | 6 | ADLITTLE, AFIRO, and KB2, each at 0.1% and 1% contamination |

The challenge set was assembled using output-informed family and representation selection. The model outputs informed choices about representation mismatch, coder confounding, and memory coverage; the resulting set may favor the applicability of the supplied memory. Reporting also involved output-informed run selection. Results on this exploratory set are descriptive and should be accompanied by all-run reporting. Independent confirmation requires a separately frozen benchmark and a prospectively fixed protocol.

Publication and benchmark references:

1. A. Ben-Tal, L. El Ghaoui, and A. Nemirovski. *Robust Optimization*. Princeton University Press, 2009. Examples 1.1.1 and 2.3.6. [Book](https://www2.isye.gatech.edu/~nemirovs/ROBook.pdf).
2. Z. Li, R. Ding, and C. A. Floudas. *A Comparative Theoretical and Computational Study on Robust Counterpart Optimization: I. Robust Linear Optimization and Robust Mixed Integer Linear Optimization*. Industrial & Engineering Chemistry Research, 50(18):10567--10603, 2011. [DOI: 10.1021/ie200150p](https://doi.org/10.1021/ie200150p).
3. Z. Li, Q. Tang, and C. A. Floudas. *A Comparative Theoretical and Computational Study on Robust Counterpart Optimization: II. Probabilistic Guarantees on Constraint Satisfaction*. Industrial & Engineering Chemistry Research, 51(19):6769--6788, 2012. Section 5.1, equations (5.1)--(5.2), Tables 5.1--5.2. [DOI: 10.1021/ie201651s](https://doi.org/10.1021/ie201651s).
4. M. Goerigk and M. Khosravi. *Benchmarking Problems for Robust Discrete Optimization*. Computers & Operations Research, 166:106608, 2024. [DOI: 10.1016/j.cor.2024.106608](https://doi.org/10.1016/j.cor.2024.106608). [Author benchmark site](https://robust-optimization.com/). Official archive URLs and instance filenames are retained in each applicable record's `source` field.
5. A. Ben-Tal and A. Nemirovski. *Robust solutions of Linear Programming problems contaminated with uncertain data*. Mathematical Programming, 88:411--424, 2000. [DOI: 10.1007/s101070000163](https://doi.org/10.1007/s101070000163). [NETLIB LP models](https://www.netlib.org/lp/data/).

The `description`, `optimal_value`, and reference `solution` values are retained unchanged from the evaluated data. Record identifiers in `external.json` are normalized for this release; `source_problem_id` and `source` retain the publication-level correspondence. Unrelated build and model-output audit metadata are omitted.
