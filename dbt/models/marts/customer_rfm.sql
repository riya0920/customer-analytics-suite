-- The RFM mart: one row per customer, calibration window only.
--
-- The calibration cutoff lives in ONE place -- a variable -- rather than being
-- retyped in every model. A holdout boundary that appears in four files is a
-- holdout boundary that will eventually differ between two of them, and the
-- resulting leakage is invisible in every individual query.
with t as (
    select * from {{ ref('stg_transactions') }}
    where t_days <= {{ var('calibration_days') }}
)
select
    customer_id,
    count(*)                                  as frequency,
    max(t_days)                               as recency_day,
    {{ var('calibration_days') }} - max(t_days) as days_since_last,
    min(t_days)                               as first_purchase_day,
    avg(order_value)                          as avg_order_value,
    sum(order_value)                          as total_value,
    avg(n_categories)                         as avg_categories,
    avg(case when used_discount then 1.0 else 0.0 end) as discount_rate
from t
group by customer_id
