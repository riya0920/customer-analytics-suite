-- Asserts the property the model description claims: crediting every touching
-- channel makes the total EXCEED the true conversion count. If this ever stops
-- being true, somebody has quietly turned a reach table into an attribution one.
with total as (
  select sum(touches_on_converting_journeys) as credited from {{ ref('channel_daily') }}
), actual as (
  select count(distinct journey_id) as converted
  from {{ ref('stg_touches') }} where converted
)
select * from total, actual where credited <= converted
