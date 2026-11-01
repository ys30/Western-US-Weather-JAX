# Figure 3 bootstrap verification

Verified from downloaded artifacts of successful CI runs:
- Adapted frozen transfer: 41274429714, artifact 13071375216, `frozen-southwest/seed-{42,7,123}.json`.
- Source-only: 41277714617, artifact 13071868167, `seed-{42,7,123}.json`.

Source JSON entries: `fixed_heat_paired_block_sensitivity` or `fixed_hot_paired_block_sensitivity`, selecting `block_steps == 56` (14 days at 4 six-hourly samples per day). Each used 1000 successful bootstrap draws. Adapted point estimates are `difference_kelvin`; bounds are `ci95_lower_kelvin` and `ci95_upper_kelvin`.

**Audit outcome:** Previously committed interval endpoints were correct for all six observations, but Figure 3 incorrectly placed circular dots at interval midpoints. It has now been corrected: dots represent source JSON `difference_kelvin`, while horizontal lines represent original percentile bounds. The CSV adds the point estimate; all values remain unrounded.
