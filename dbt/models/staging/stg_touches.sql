-- One row per TOUCH, exploded from the journey arrays.
--
-- The explode belongs here rather than in Python because it is the point at
-- which a nested structure becomes a table, and every consumer downstream wants
-- the table. Doing it in three different notebooks is how three different touch
-- counts end up in three different decks.
select
    cast(journey_id  as integer) as journey_id,
    cast(customer_id as integer) as customer_id,
    cast(journey_index as integer) as journey_index,
    cast(position    as integer) as touch_position,
    cast(channel     as varchar) as channel,
    cast(touch_day   as double)  as touch_day,
    cast(converted   as boolean) as converted
from {{ source('raw', 'touches') }}
