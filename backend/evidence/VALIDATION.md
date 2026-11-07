# FloodGuard validation

What was tested, what it showed, and what it cannot show. Reproduce with `python -m scripts.backtest` and
`python -m scripts.site_check`. Full numbers are in `backtest_results.json` and `site_check.json`.

## Summary

- **Rain alerts have real skill.** A 20 mm/24 h trigger on forecasts as issued at the time caught 6 of 7 documented
  Nairobi floods. Chance would catch about 24% of them with the same number of alert days.
- **The sample is small.** With 7 events, the 95% interval on 6 of 7 is roughly 49% to 97%.
- **Road susceptibility does not predict where floods happened.** 5 of 8 reported flood sites rate Low, so it is used
  only to raise urgency and can never silence an alert.
- **False alarms are high.** At most about 1 in 6 alert episodes matched a catalogued flood. That is a lower bound
  because the catalogue is incomplete, and each of these days was still forecast heavy rain.

## Rainfall backtest

**Period:** 2024-02-01 to 2026-11-05 (1,009 days). **Events:** 7 dated Nairobi floods in `flood_events.csv`.

**Method:** Open-Meteo Previous Runs data replays what forecasts said before each date, so results do not benefit from
hindsight. An event counts as detected if an alert fired on the event dates or the day before. River discharge is not
tested because past forecasts of it are not available.

City-wide, alerting if any zone crosses the threshold (short-lead forecast):

| Trigger, mm/24 h | Events detected | Expected by chance | Alert days per year | Alert episodes | Episodes matching an event |
|---|---|---|---|---|---|
| 10 | 6/7 | 51% | 86 | 61 | 5 |
| **20** | **6/7** | **23%** | **34** | **36** | **6** |
| 30 | 2/7 | 7% | 9 | 14 | 3 |
| 50 | 1/7 | 1% | 1.4 | 2 | 1 |

Day-ahead forecasts at 20 mm also detected 6 of 7 (36 alert days per year). The 50 mm trigger alone would have missed
six of the seven floods, because model rain during them mostly peaked at 20 to 45 mm. This is why the default alert
level is Watch, which covers the 20 mm band, rather than Warning.

### What one subscriber would see

The table above is optimistic for an individual, who is alerted only for their area. At 20 mm:

| Rain considered | Short-lead: events detected | Day-ahead: events detected | Alert days per year | Episodes per year |
|---|---|---|---|---|
| Own zone only | 3.0/7 | 5.0/7 | 9 to 15 | 5 to 7 |
| Any zone within 10 km | 4.8/7 | 5.9/7 | 18 to 23 | 9 |
| **Any zone within 15 km (used)** | **5.6/7** | **6.0/7** | **23 to 27** | **10** |
| Whole city | 6.0/7 | 6.0/7 | 34 to 36 | 12 to 13 |

Figures are averages across the 31 zones. The worst zone at 15 km detected 4 of 7 on short-lead forecasts.

### Known miss

E5 (15 March 2026) reached only 7 to 12 mm in forecasts and 4 mm in ERA5 reanalysis, so no sensible threshold catches
it. Localised storms can fall between grid cells. The Kenya Met feed and future gauge or radar data are the mitigation.

The 20 mm threshold was chosen a priori from Kenya Met advisories. The neighbourhood radius and default alert level
were chosen after seeing these same events, so those per-subscriber numbers are likely somewhat optimistic.

## Road susceptibility check

Eight places named in the flood reports were scored against the city-wide distribution.

| Site | Tier | City percentile |
|---|---|---|
| Mathare 4A | High | 86 |
| Mukuru kwa Reuben | High | 81 |
| Utawala | Medium | 66 |
| Likoni Bridge, South B | Low | 41 |
| Kariokor | Low | 33 |
| Seasons Road | Low | 33 |
| Jogoo Road | Low | 33 |
| Ruai bypass at Kangundo Road | Low | 13 |

The median site sits at the 37th percentile, below a typical location. Coordinates are approximate, and results are
sensitive to them: a point 150 m from the Mathare 4A site rates Low. The model's held-out PR-AUC of 0.578 measures
agreement with FastFlood simulations, not real floods, and FastFlood does not model drainage, which is a likely cause
of urban road flooding.

## Software testing

`pytest` runs 100+ tests with no network access. They cover the alert rules, SMS length limits, sign-up abuse cases,
and failure paths: forecast, river, KMD and SMS outages, stale data, and the spending cap. A live run processed
31 of 31 zones from the real APIs with no source errors. Real SMS delivery has not been tested.

## Limitations

- **Event catalogue:** 7 events from news reports, with no record of dry days or minor floods. Only two dates come
  from articles read in full; the rest come from search snippets or URL dates (see `date_evidence`).
- **Coverage:** Nairobi County only. The model and road data do not extend beyond it.
- **Time-of-day:** the backtest scores by calendar day, so it does not measure warning lead time in hours.
- **Open-Meteo licence:** the free API is for non-commercial use. A commercial launch needs a paid plan.
