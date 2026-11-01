# Model identity and experiment dates

## Model roles in the paper

| Role | Reported model | Requested API model name |
| --- | --- | --- |
| Reformulator, reflector, and application formulator | DeepSeek-V4-Flash | `deepseek-v4-flash` |
| Fixed coder | DeepSeek-V4-Pro | `deepseek-v4-pro` |

The primary experiments took place during the April 2026 DeepSeek-V4 Preview release period. These names identify model families and API aliases. The archived experiments did not retain an immutable model revision, API-returned deployment identifier, or system fingerprint, so an exact historical backend snapshot cannot be recovered from the saved results.

## Archived experiment periods

| Experiment phase | Recorded dates | Evidence and scope |
| --- | --- | --- |
| Primary memory adaptation | 2026-05-01 to 2026-05-02 | Training log explicitly records the start and end; the supplied eight-experience memory matches this run. |
| Main reformulation evaluations | 2026-05-02 to 2026-05-04 | Archived Random, Hard, and Large run labels. Some baseline reformulations were generated on 2026-04-29 and recoded on 2026-05-02. |
| Application evaluation | Early May 2026 | An archived 2026-05-05 run matches the reported 81.3% accuracy; a complete source manifest for the plotted comparison was not retained. |
| General-P and Coupled-Row supplemental evaluations | 2026-07-26 | Supplemental experiment records. |
| Independent-memory re-evaluations | 2026-08-03 to 2026-08-04 | These are inference dates. The corresponding additional memories were trained on 2026-05-03 to 2026-05-04. |
| ExternalRO32 supplemental evaluations | 2026-08-04 | Archived external evaluation records. |

Historical dates are transcribed from training logs and automatically generated run labels, not filesystem modification times. The original logs did not declare a timezone. Supplemental records preserve the requested Flash/Pro aliases, but do not establish an immutable backend revision; a specific served revision cannot be inferred from a date alone.

The requested aliases were reused across these periods, so the served revisions can differ between experimental phases.

## API aliases can change

DeepSeek's [April 24, 2026 announcement](https://api-docs.deepseek.com/news/news260424/) identifies the V4 Preview release. Its [official change log](https://api-docs.deepseek.com/updates/) records a Flash update on July 31, a Pro update on August 13, and routing of the older `deepseek-v4-flash` name to V4.1 Flash on September 10. These documentation entries were checked on 2026-11-01.

The requested names in this repository preserve the paper's API configuration. Calling those names today does not pin the weights used for the paper. Later runs can differ because of deployment updates as well as inference nondeterminism. A dated release label in an announcement should not be assumed to be a callable, pinned API model ID.

## Metadata recorded by new runs

Training and evaluation write `run_metadata.json` with UTC start/end timestamps, completion status, requested model names, and non-sensitive request settings. They also write `base_api_calls.jsonl` and `coder_api_calls.jsonl` for successful API responses. These records contain request/response UTC timestamps, the requested model name, and the returned model name, response creation time, and system fingerprint when the API provides them. Unavailable fields remain `null`.

These metadata files exclude API keys, authorization headers, prompts, full API responses, and endpoint URLs. A returned model name can itself be an alias; logging it improves traceability without guaranteeing an immutable snapshot. The output directory is excluded from Git by default.

## Suggested manuscript statement

The main experiments were conducted in late April and early May 2026 using DeepSeek-V4-Flash for reformulation, reflection, and application formulation, and DeepSeek-V4-Pro as the fixed coder. The requested API model names were `deepseek-v4-flash` and `deepseek-v4-pro`, during the DeepSeek-V4 Preview release period. The primary memory was adapted on May 1--2, 2026. Supplemental evaluations were conducted in July and August 2026. The historical records identify request aliases and execution dates but do not retain immutable backend revision identifiers; later API calls using the same aliases may access updated deployments.
