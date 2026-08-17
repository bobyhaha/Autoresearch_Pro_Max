# Scientific method and confidence policy

This document defines the `confidence-v1` scientific contract. Changes to scoring semantics
require a new policy version; historical records retain their original assessment version.

## Epistemic objects

A source is not a claim, a claim is not a mechanism, and a mechanism is not a confirmed
hypothesis.

```text
ResearchAgenda
    -> LiteratureSearch
        -> LiteratureSource snapshot
            -> ScientificClaim (supports or opposes one belief key)
                -> ScientificMechanism edge
                    -> ScientificHypothesis
                        -> ExperimentSpec / EvidenceDecision
                            -> experiment-origin ScientificClaim
```

Each arrow is an explicit immutable identifier. Views may be deleted and rebuilt without
losing this lineage.

## Atomic claims

One claim states one scoped proposition. It records:

- a stable `belief_key` grouping semantically equivalent support and opposition;
- whether it supports or opposes that proposition;
- source/evidence/derived-claim identifiers;
- an exact paper locator and short attributable excerpt when applicable;
- the study design, artifacts, reproduction state, directness, scope transfer, and bias;
- reported metrics including comparator, effect, uncertainty, and seeds in structured text;
- the assessor identity, time, rationale, and confidence policy version.

An agent must not merge several paper conclusions into one extraction, infer a mechanism
from a benchmark win without marking it indirect/speculative, or copy the abstract as if it
were a methods audit.

## Literature contribution

For one claim/source pair, component scores are combined as:

```text
quality =
    0.06 * venue
  + 0.02 * peer_review
  + 0.18 * study_design
  + 0.08 * content_depth
  + 0.12 * artifact
  + 0.22 * reproduction
  + 0.14 * directness
  + 0.12 * scope_match
  + 0.06 * risk_of_bias
```

The weights sum to one. Venue prestige is intentionally small; reproduction is more than
three times as influential. Component mappings are:

| Component | Values from strongest to weakest |
|---|---|
| venue | top .90; selective .82; peer reviewed .72; workshop .60; preprint .46; unknown .42 |
| peer review | yes .80; unknown .45; no .38 |
| study design | meta-analysis .95; randomized .92; controlled benchmark .82; ablation .76; observational/theoretical .58; case study .48; unknown .40; anecdotal .22 |
| content depth | full-text snapshot .95; abstract only .55; metadata only .25 |
| artifact | verified .95; available .76; partial .58; unknown .42; none .35 |
| reproduction | independent success .98; author only .65; unknown/not attempted .42; independent failure .05 |
| directness | direct .95; indirect .62; speculative .30 |
| scope match | exact 1.00; close .82; partial .58; unknown .42; distant .30 |
| risk of bias | low .92; medium .64; unknown .46; high .28 |

A normal study contributes signed evidence mass `1.45 * quality`; a meta-analysis contributes
`1.90 * quality`. Support is positive and opposition is negative. A declared independent
reproduction failure reverses the direction so a paper cannot label failed reproduction as
support.

Repeated source snapshots and repeated extractions are correlated. For each stable
`work_key` and stance, only the strongest contribution enters the aggregate.

## Belief confidence

For independent contribution masses `m_i`:

```text
confidence = sigmoid(sum(m_i))
```

This starts an unobserved belief at 0.5. It is a transparent heuristic evidence confidence,
not a frequency-calibrated probability. Labels are:

- below .20: `very_low`;
- .20–.39: `low`;
- .40–.59: `uncertain`;
- .60–.74: `moderate`;
- .75–.89: `high`; and
- .90 or greater: `very_high`.

The state is `supported` at confidence >= .70, `rejected` at <= .30, and otherwise
`uncertain`. If independent support and opposition each have at least .45 evidence mass,
the state is `contested` regardless of the arithmetic winner. The view always publishes
every component and contribution.

## Experimental contribution

Only valid, claim-eligible evidence with a numeric preregistered effect enters an
experiment-origin claim. Evidence is grouped at the ExperimentSpec level so five seeds do
not masquerade as five independent studies.

- A complete confirmation receives reliability `min(.98, .90 + .02*n_valid)`.
- A pilot receives reliability `.30` and remains exploratory.
- An incomplete confirmation receives at most `.25` and is not decisive.
- The preregistered success rule determines support; the falsifier rule determines
  opposition; an inconclusive aggregate contributes no directional evidence.
- A decisive experiment contributes signed mass `2.2 * reliability`.

The executed result determines direction. If a manually declared stance disagrees, the
view reports `declared_stance_mismatch` and does not let the label reverse the evidence.

## Mechanisms and hypotheses

A mechanism is a causal graph, not a prose field. Every edge cites supporting claims.
Opposing claims belong in the belief ledger and alternative explanations. Mechanism
confidence equals the weakest edge confidence, because a causal chain is no stronger than
its least supported necessary link.

A hypothesis cites claims and mechanisms, states a benchmark scope, makes a directional
metric prediction, names a minimum effect when relevant, lists falsifiers and competing
hypotheses, and proposes one minimal intervention plus diagnostic measurements.

Default model-search readiness requires:

- every hypothesis topic belongs to an agenda;
- agenda source/search/claim/full-text requirements are complete;
- at least two independent foundation evidence units;
- foundation confidence >= .60; and
- a valid mechanism graph.

The explicit `--allow-weak-science` override supports novel high-risk tests. It does not
alter confidence or remove the immutable hypothesis link.

## Continuous refinement

`RESEARCH_TASKS.json` is the agent work queue. It derives tasks to:

1. search sparse, stale, or sufficiently old contested topics;
2. obtain/read full text and extract claims;
3. resolve independent contradictions;
4. strengthen or discriminate weak mechanism edges;
5. stage research-ready hypotheses;
6. refine preliminary, inconclusive, or contested hypotheses; and
7. materialize decisive confirmations as experiment-origin claims.

These are CPU/reasoning tasks. They may run in another process while `run --follow` keeps
GPU workers occupied. Belief scoring itself performs no network, model, embedding, or GPU
call.

## Known limitations

- The confidence number is not empirically calibrated against a truth set.
- Venue classification is a small versioned ML-oriented mapping; unrecognized venues stay
  unknown and should be curated explicitly.
- Bibliographic metadata cannot prove peer review, artifact correctness, or reproduction.
- Semantic equivalence is asserted through `belief_key` and must be audited by the agent.
- Correlation across different papers from the same lab/dataset is not yet modeled beyond
  stable-work deduplication.
- OpenAlex retrieval supplies metadata and abstracts. Full-text reading and claim extraction
  remain agent work with immutable snapshots and locators.

These limits are published rather than hidden because a well-defined scientific system
must expose what its confidence score does not know.
