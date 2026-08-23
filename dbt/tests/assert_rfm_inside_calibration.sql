-- Fails if any RFM row was built from a transaction after the cutoff.
-- A singular test rather than a dbt_utils generic one, because dbt_utils cannot
-- be fetched in this environment -- and a package dependency that silently does
-- not resolve turns every test into a passing no-op.
select customer_id
from {{ ref('customer_rfm') }}
where recency_day > {{ var('calibration_days') }}
