-- Channel x day touch and conversion counts: the grain a marketing dashboard
-- reads, and the grain attribution should NOT read.
--
-- A conversion here is credited to every channel that touched the journey, so
-- the column sums to more than total conversions. That is stated in the schema
-- test rather than hidden: it is a REACH table, not an attribution table, and
-- the single most common analytics error in this domain is treating one as the
-- other.
select
    channel,
    cast(floor(touch_day) as integer) as day,
    count(*)                          as touches,
    count(distinct journey_id)        as journeys_touched,
    sum(case when converted then 1 else 0 end) as touches_on_converting_journeys
from {{ ref('stg_touches') }}
group by channel, cast(floor(touch_day) as integer)
