-- One row per transaction, typed and named.
--
-- Staging does exactly one thing: rename and cast. No business logic, no joins,
-- no filtering. That discipline is what makes the marts layer reviewable -- if
-- staging is allowed to filter, every downstream number silently depends on a
-- WHERE clause nobody reads.
select
    cast(customer_id as integer)       as customer_id,
    cast(t_days      as double)        as t_days,
    cast(order_value as double)        as order_value,
    cast(n_categories as integer)      as n_categories,
    cast(used_discount as boolean)     as used_discount
from {{ source('raw', 'transactions') }}
