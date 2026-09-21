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
    cast(n_products as integer)      as n_products,
    cast(had_return as boolean)     as had_return
from {{ source('raw', 'transactions') }}
