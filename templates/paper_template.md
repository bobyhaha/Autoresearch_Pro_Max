# Paper title

Version: v1
Date:
Comparison protocol:
Covered ExperimentSpecs:
Covered EvidenceDecisions:

## Abstract

Briefly state:

- the problem;
- mechanisms studied;
- number of pilot and confirmation experiments;
- principal result;
- strongest negative finding;
- SOTA decision.

## 1. Research question

Describe the unresolved question and why it matters.

## 2. Prior knowledge

List the observations, previous experiments, papers, and beliefs that
motivated the research direction.

Clearly distinguish:

- established facts;
- prior experimental observations;
- external claims;
- assumptions;
- speculation.

## 3. Proposed mechanisms

### Mechanism A

Cause:
Causal chain:
Predicted mediator:
Predicted endpoint:
Alternative explanations:

### Mechanism B

...

## 4. Registered hypotheses and falsifiers

| Spec | Hypothesis | Prediction | Falsifier | Stage |
|---|---|---|---|---|
| exp_a | ... | effect < -0.01 | effect >= 0 | confirmation |

## 5. Experimental method

Describe:

- model and training budget;
- dataset and tokenizer;
- comparison group;
- metric;
- control and candidate arms;
- replicate plan;
- GPU isolation policy;
- code and data bindings;
- admissibility requirements;
- registered aggregation rules.

## 6. Results

Include every relevant result, not only improvements.

| Spec | Replicates | Effect mean | Stdev | Evidence | Conclusion |
|---|---:|---:|---:|---|---|
| exp_a | 3/3 | -0.012 | 0.003 | valid | supported |
| exp_b | 2/3 | — | — | preliminary | insufficient |
| exp_c | 0/1 | — | — | invalid | instrumentation failure |

## 7. Mechanism assessment

For each mechanism:

- Was the registered prediction observed?
- Was the falsifier triggered?
- Was the predicted mediator measured?
- Could the result be explained by throughput, parameter count, or another confound?
- Does the result support the mechanism or merely the intervention?

A better endpoint metric alone should not automatically be described as mechanism confirmation.

## 8. Negative findings and failures

Report:

- regressions;
- crashes;
- OOMs;
- insufficient-step runs;
- hash or telemetry failures;
- rejected hypotheses;
- interventions that were equivalent to control.

## 9. Belief revisions

State explicitly:

- previous belief;
- new belief;
- supporting EvidenceDecision IDs;
- confidence or remaining uncertainty;
- what evidence would reverse the revision.

## 10. SOTA decision

Report:

- previous verified SOTA;
- candidate aggregate value;
- comparison group;
- replicate values;
- promotion or non-promotion decision;
- code snapshot hashes;
- data-manifest hash.

## 11. Limitations

Examples:

- single hardware platform;
- one-run experiments;
- small sample size;
- unmeasured mediator;
- fixed five-minute budget;
- possible adaptive-search bias;
- conclusions that do not generalize beyond the comparison group.

## 12. Recommended next experiments

For each proposed experiment:

- mechanism;
- expected information gain;
- prediction;
- falsifier;
- required stage;
- approximate compute cost.

## Reproducibility appendix

List:

- ExperimentSpec IDs and digests;
- ExecutionManifest IDs;
- ResultBundle IDs;
- EvidenceDecision IDs;
- exact argv;
- code SHA-256;
- data SHA-256;
- execution hosts and GPUs;
- environment information;
- artifact paths and hashes.
