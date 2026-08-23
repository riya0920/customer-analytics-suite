-- What each customer actually did AFTER the cutoff. The label, kept in its own
-- model so that nothing which builds features can accidentally select from it.
select
    customer_id,
    count(*)          as holdout_orders,
    sum(order_value)  as holdout_value,
    max(t_days)       as holdout_last_day
from {{ ref('stg_transactions') }}
where t_days > {{ var('calibration_days') }}
group by customer_id
