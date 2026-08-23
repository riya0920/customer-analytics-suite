-- The leakage guard. If this ever returns rows, features and labels overlap.
select customer_id
from {{ ref('customer_holdout') }}
where holdout_last_day <= {{ var('calibration_days') }}
