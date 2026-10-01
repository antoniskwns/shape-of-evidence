# The Shape of Evidence: Fuzzy Prediction Sets for Downstream Decisions

MSc Quantitative Finance thesis, Erasmus School of Economics, 2026.
Supervisor: Dr. Nick Koning.

The thesis is [`paper/The Shape of Evidence.pdf`](paper/The%20Shape%20of%20Evidence.pdf).

## What it is about

A decision-maker asks a statistician for evidence about an uncertain outcome,
in the form of an e-value for every outcome. Many valid e-values exist, and
they can lead to different actions. The standard choice, the likelihood ratio,
is the most informative one but ignores what a wrong action costs. The thesis
asks how the evidence should be designed when the decision-maker's loss
function is known in advance.

- **Result 1.** For a chosen action, the profile proportional to that action's
  loss gives the smallest possible bound on its expected loss, and it is the
  only profile that does.
- **Result 2.** That profile leads to the action it protects, but the most
  informative profile need not, even when the action is the one the belief
  itself recommends. Keeping as much information as possible while reaching
  the action means imposing it as a constraint, and the optimum then sits
  where the action ties with a competitor.
- **Result 3.** A reshaped likelihood ratio can reproduce the loss-based
  profile only when the loss is itself a function of the ratio.

The results are applied to a shopkeeper ordering replica shirts before the
2026 World Cup final, using Polymarket prices as the forecast, and tested in
controlled simulations.

## Repository layout

```
paper/   LaTeX source, bibliography and the compiled thesis
code/    decision_pipeline.py: every number and figure in the thesis
         extract_market_tape.py: builds the price series from the order book
         data/: Polymarket order-book files and the derived price series
         figures/, results.txt: outputs of decision_pipeline.py
```

## Reproducing the thesis

```
pip install -r code/requirements.txt
make
```

`make` runs `code/decision_pipeline.py`, which writes the figures and
`code/results.txt` in about ten seconds, and then compiles the thesis with
`latexmk`. Everything is exact except the random-shape search of Section 4.4,
which uses a fixed seed, so every run gives the same output.

The price series `code/data/polymarket_match_odds.csv` is included. To rebuild
it from the raw order-book files, run `make tape` (needs `pandas` and
`pyarrow`).

## Data

The order-book data come from the Marketlens dataset *Polymarket Limit Order
Book: The 2026 World Cup Final, All 323 Markets*
([doi:10.5281/zenodo.21933086](https://doi.org/10.5281/zenodo.21933086)),
licensed CC BY 4.0. The loss table and the model in Section 4 are modelling
choices stated in the thesis, not observed quantities.

## Licence

The code is released under the MIT licence. The thesis text and figures are
© 2026 Antonis Konstantinidis.
