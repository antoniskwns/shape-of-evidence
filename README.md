# The Shape of Evidence: Fuzzy Prediction Sets for Downstream Decisions

MSc Quantitative Finance thesis, Erasmus School of Economics, Erasmus University
Rotterdam. Supervisor: Dr. Nick Koning. Submitted and defended in October 2026.

**Read the thesis:** [`paper/The Shape of Evidence.pdf`](paper/The%20Shape%20of%20Evidence.pdf)

## What it is about

A decision-maker asks a statistician for evidence about an uncertain outcome, in
the form of an e-value for every outcome. Many valid e-values exist, and they can
lead to different actions. The standard choice, the likelihood ratio, is the most
informative one, but it ignores what a wrong action costs. The thesis asks how the
evidence should be designed when the decision-maker's loss function is known in
advance.

- **Result 1.** For a chosen action, the e-value proportional to that action's
  loss gives the smallest possible bound on its expected loss, and it is the only
  one that does. It leads to that action unless another action is cheaper in
  every outcome.
- **Result 2.** The most informative e-value need not lead to the chosen action,
  even when that action is the one the decision-maker's own forecast recommends.
  Keeping as much information as possible while reaching the action means
  imposing it as a constraint, and the optimum then sits where the action ties
  with a competitor.
- **Result 3.** Reshaping the likelihood ratio reproduces the loss-based e-value
  only when the loss is itself a function of the ratio.

The results are illustrated on a shopkeeper ordering replica shirts before the
2026 World Cup final, with Polymarket prices as the forecast, and in controlled
simulations with a known truth.

## Repository layout

```
paper/                   the thesis
├── main.tex             entry point; preamble.tex holds the formatting
├── chapters/            one file per section, 00_abstract to 07_notation
├── assets/              title-page logo
├── bibliography.bib
└── The Shape of Evidence.pdf
analysis/                everything that produces the numbers and figures
├── code/
│   ├── decision_pipeline.py     every number, table and figure in the thesis
│   ├── extract_market_tape.py   builds the price series from the order book
│   └── requirements.txt
├── data/
│   ├── raw/             Polymarket order-book files (Marketlens dataset)
│   └── processed/       the one-minute price series used by the pipeline
└── results/
    ├── figures/         the five figures included in the thesis
    └── results.txt      every number reported in the thesis
```

## Reproducing the thesis

Requirements: Python 3 and a LaTeX distribution with `latexmk`. Verified with
Python 3.13, NumPy 2.4, Matplotlib 3.10 and TeX Live 2024.

```
pip install -r analysis/code/requirements.txt
make
```

| Command        | What it does |
|----------------|--------------|
| `make`         | runs the pipeline, then compiles the thesis |
| `make figures` | runs `analysis/code/decision_pipeline.py` only (about ten seconds) |
| `make tape`    | rebuilds the price series from `analysis/data/raw/` (needs `pandas` and `pyarrow`) |
| `make clean`   | removes LaTeX build files, keeping the compiled thesis |

Every result is computed exactly, except the random-shape search of Section 4.4,
which uses a fixed seed, so every run gives the same output. The processed price
series is included, so `make tape` is only needed to check it against the raw
data.

## Data

The order-book data come from the Marketlens dataset *Polymarket Limit Order Book:
The 2026 World Cup Final, All 323 Markets*
([doi:10.5281/zenodo.21933086](https://doi.org/10.5281/zenodo.21933086)), licensed
CC BY 4.0. The loss table and the model in Section 4 are modelling choices stated
in the thesis, not observed quantities.

## Citation

```bibtex
@mastersthesis{Konstantinidis2026,
  author = {Konstantinidis, Antonis},
  title  = {The Shape of Evidence: Fuzzy Prediction Sets for Downstream Decisions},
  school = {Erasmus School of Economics, Erasmus University Rotterdam},
  type   = {{MSc} thesis},
  year   = {2026}
}
```

## Licence

The code is released under the [MIT licence](LICENSE). The thesis text and figures
are © 2026 Antonis Konstantinidis.
