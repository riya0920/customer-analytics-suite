TO:      CMO
FROM:    Customer Analytics
RE:      Marketing budget allocation, and one channel we should test

RECOMMENDATION

1. Stop allocating on last-touch. On our data it sends 20.0% of the
   budget -- about $299 of $1500 -- to retargeting, and our best
   evidence is that retargeting causes approximately none of the
   conversions it is credited with.

2. Fund a geo holdout on retargeting. Switch it off in matched control
   markets for four weeks. If the effect is real we lose a month of
   incremental conversions in half our markets; if it is not, the test
   pays for itself immediately and permanently. That asymmetry is the
   argument: the experiment is cheapest in exactly the world where the
   channel is worthless.

3. Reallocate toward paid search and email, which carry 37% of the
   measurable effect between them.

WHAT THIS IS BASED ON

We simulated marketing journeys with KNOWN channel effects and scored
every standard attribution method against that truth. Under those
conditions:

  - Every method credits a channel we know causes nothing. Last-touch
    gives it 20% of all credit; even Shapley, which is designed to
    give a useless channel exactly zero, gives it 11%.
  - The reason is not the estimators. It is that retargeting is TARGETED
    at customers who were already going to buy, so it correlates with
    conversion without causing it. No amount of modelling separates
    correlation from causation in data that contains no experiment.
  - Allocating on last-touch instead of truth costs 358 conversions
    (12.9% of achievable) on a $1500 budget.

WHAT WE ARE NOT CLAIMING

  - These are simulated channel effects, not measured ones. What
    transfers is the RANKING of methods and the size of the error they
    make, not the specific percentages.
  - Our CLV model ranks customers well and mispredicts individuals. Use
    it to size a segment, never to decide what one customer is worth.
  - The channel-value weighting in section 6 is correlational and
    inherits the same confound. It sharpens the case for the experiment;
    it does not substitute for it.

COST OF DOING NOTHING

  Roughly $3592 a year of budget flowing to a channel whose effect we
  have never measured, and a reported ROAS that will keep telling us it
  is working, because a channel that follows intent always looks good to
  a correlational metric.
