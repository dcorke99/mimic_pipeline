# Semi-synthetic endpoint note

The semi-synthetic generator originally allowed repeated CAUTI events because it did not end the CAUTI risk set after the first event. It now retains the first CAUTI and clears all later events and risk periods.

A second inconsistency occurred when the simulated catheter was never removed but the original episode ended in reinsertion. Because reinsertion is then impossible, the episode now ends at the same time as a `simulation_horizon`. This retains the full cohort without inventing unobserved clinical history after the original endpoint.

This is acceptable for the PhD's main purpose: validating Policy-DML and LTMLE against known policy values under a common semi-synthetic data-generating process. It does not model how catheter policy might change discharge, death, reinsertion, or later outcomes. The semi-synthetic analyses therefore use a supported 14-day horizon and should be described as method validation under a fixed observed follow-up structure, rather than as a complete clinical simulation.

Consider later running a sensitivity analysis that excludes episodes whose original endpoint was reinsertion. This is sufficient to check whether the validation conclusions depend materially on the chosen endpoint treatment; building a complete longitudinal ICU simulator is not currently necessary.
