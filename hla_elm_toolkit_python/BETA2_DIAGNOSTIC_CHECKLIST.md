# Β.2 diagnostic checklist — Section 3.11 identical n=45 across cohorts

> **Status: RESOLVED.** This issue was confirmed to be a copy-paste
> transcription error made while writing up Section 3.11, not a bug in
> the evaluation code. The correct per-cohort evaluable counts are
> n=45 (GRPT, unchanged), n=38 (ORAM), and n=29 (HTO), with accuracy
> remaining 100.0% in each case; the article (Section 3.11, Table 19,
> Figure S15) has been corrected accordingly. This checklist is kept
> below as a general debugging reference, since the same kind of
> symptom (suspiciously identical figures across differently-sized
> cohorts) is worth checking for elsewhere, and step 1 below (checking
> for a write-up transcription error before auditing the code) is what
> ultimately identified the actual cause in this case.

## The symptom

Section 3.11 (the in-house Hapl-o-Mat-style five-locus EM baseline)
reports, for the joint (all-evaluable-loci) accuracy row of Table 19:

| Cohort | n | Joint accuracy |
|---|---|---|
| GRPT (n=4,353 donors) | 45 | 100.0% |
| ORAM (n=20,086 donors) | 45 | 100.0% |
| HTO (n=117,345 donors) | 45 | 100.0% |

Three cohorts differing in size by up to **27x** produce the *exact
same* evaluable-donor count and the *exact same* joint accuracy. This is
possible in principle (e.g. if all three cohorts happen to share an
identical small subpopulation with complete confirmatory typing at every
locus) but is very unlikely, and warrants a direct check before the
figures are relied upon.

## Most likely cause

A donor-filtering variable (the set of "donors with confirmatory
two-field ground truth at all five loci simultaneously") computed once
for the **first** cohort scored (most likely GRPT, since Section 3.11's
narrative describes it first) and then **not recomputed** when scoring
subsequently switched to ORAM and HTO — e.g. a variable such as
`evaluable_donors` or `joint_eval_set` that was assigned outside the
per-cohort loop, or a notebook cell that was not re-run for the later
cohorts before their results were captured.

## Step-by-step check

1. **Locate the exact function/cell that produces the "n evaluable
   (joint)" and "Joint accuracy" numbers for Section 3.11.** Confirm
   there are three separate calls (or three separate notebook executions)
   — one per cohort — rather than one call whose output was copied three
   times.

2. **Print the donor-ID set used for the joint-accuracy denominator in
   each of the three runs**, e.g.:
   ```python
   print(cohort_name, len(evaluable_donor_ids), sorted(evaluable_donor_ids)[:5])
   ```
   If the three printed sets are identical (same IDs, not just same
   count), that confirms the filtering variable was never recomputed —
   the bug is upstream of the accuracy calculation itself.

3. **If the sets differ but the count and accuracy still coincide**,
   check whether the *donors* differ but the underlying *haplotype
   pool* used to determine "evaluable" is shared/cached across cohorts
   (e.g. a `candidate_haplotypes` object built once from GRPT and reused
   for ORAM/HTO scoring without rebuilding it against each cohort's own
   typed loci).

4. **Check whether cohort-selection logic uses a stale reference to the
   previous cohort's dataframe**, e.g.:
   ```python
   # bug pattern to look for:
   df = grpt_df
   for cohort_name, _ in [("GRPT", grpt_df), ("ORAM", oram_df), ("HTO", hto_df)]:
       # df was never reassigned inside the loop body
       run_scoring(df)
   ```

5. **Once the bug is found**, re-run the scoring for ORAM and HTO
   independently and update Table 19's three "Own 5-locus EM baseline"
   rows (and Figures S14-S15, which are generated from the same numbers)
   with the corrected, cohort-specific n and accuracy values.

6. **If, after this check, the three cohorts genuinely do produce
   identical figures** (e.g. because the evaluable subset really is
   anchored on the same small, shared, fully-typed sub-population across
   all three files — which would itself be a noteworthy and
   citable finding), replace the current unexplained coincidence with an
   explicit sentence in Section 3.11 stating and justifying this, rather
   than leaving it to look like an unexamined error.

## What this checklist does not do

It cannot identify the actual bug without access to the evaluation
script and the real donor-level data, neither of which are available in
this toolkit (see `README.md` Section 1). It is intended to make the
audit in your own codebase fast and targeted rather than open-ended.
