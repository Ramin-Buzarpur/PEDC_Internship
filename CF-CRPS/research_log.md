# EHF-CRPS Research Log

## Goal
Find a novel loss function for financial return distribution forecasting that
beats standard losses (Gaussian NLL, Student-t NLL, Gaussian CRPS) on tail
metrics AND overall CRPS, is mathematically proper (or provably a valid
scoring rule), algorithmically sound (gradient flow, no degenerate optima),
and paper-worthy.

## Prior findings (from report.txt + existing code)
- SC-twCRPS v1.0: SC_only CRPS=2.74 vs StudentT=0.778. State weighting adds
  nothing (SC_only == SC_NoState). Hybrid ~ StudentT. Dead end as main loss.
- Root cause hypotheses: (a) the SC penalty is L1-on-samples and can be gamed
  by sigma inflation in the tail; (b) rsample() for StudentT is approximate
  (normal-mean mixture) so MC noise dominates; (c) tail mask uses the
  EMPIRICAL quantile of |y| (selection on the outcome -> selection bias).
- TCM-CRPS v0.3 exists: CRPS + lambda * tail quantile exceedance L2. Proper
  for (F, q_alpha) triple. Needs benchmarking.

## Baselines to beat (synthetic GARCH-like data, identical backbone net)
- Gaussian NLL, Student-t NLL, Gaussian CRPS (analytic), plus TCM-CRPS.

## Environment
- torch 2.13.0+cu126, no GPU. Python via `python`. No pandas needed.
- NOTE: sanity_check.py imports gaussian_crps/sc_twcrps/SC_twCRPS_loss from
  SC_twCRPS.py but that file no longer defines them -> stale, do not use as is.

## Results (updated as we go)
- (pending)
