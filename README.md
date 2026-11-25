# Dynamic pricing in a Kazakhstan eSIM marketplace

The project compares **Q-learning** and **SAC** against **Fixed**, **TwoBound**, and **CartelTwoBound** competitors in a simulated market for 10 GB eSIM packages. Both use the stochastic demand model in `customer_model.py`. Market prices, exchange rate, speeds, arrival rate, and income distribution are experimental assumptions. The experiments measure each learner's revenue against its competing provider. Measuring improvement over the learner's own fixed-price strategy would require a separate fixed-vs-fixed experiment.

## Install and validate

Use 64-bit CPython 3.7. The pinned stack includes TensorFlow 1.15, Stable Baselines 2, Keras 2.2, Gym 0.17, and SciPy 1.7.3. Run the following commands from the repository root:

~~~sh
python -m pip install -r requirements.txt
python test_market.py
python run_experiments.py --smoke-test
python run_analysis.py --input-dir experiments/esim_smoke --output-dir results/esim_smoke
python plot_results.py --input-dir experiments/esim_smoke --output-dir results/esim_smoke/figures --smoothing-window 20
~~~

In PowerShell with the existing virtual environment, replace `python` with `.\.venv\Scripts\python.exe`. The smoke test runs 160 training and 20 evaluation steps per matchup and verifies that both learners update. TensorFlow 1.15 may emit deprecation and missing CUDA notices on CPU machines.

## Run the experiments

~~~sh
python run_experiments.py
python run_analysis.py
python plot_results.py
~~~

The defaults in `gym_esim.json` are five independent runs per matchup, 50,000 training steps and 10,000 evaluation steps per run. Prices range from 1,000 to 10,000 KZT. Q-learning uses a 91-point price grid and the Double, dueling, and prioritized replay settings from `gym_ql_nice.json`. SAC uses continuous actions and the network and exploration settings in `gym_sac.json`. Provider B opens at 4,000 KZT. Choose a competitor with `--competitor fixed`, `--competitor twobound`, or `--competitor cartel`. Omitting the flag runs every configured competitor.

- **Fixed** charges 4,000 KZT in both half-periods.
- **TwoBound** undercuts A by 100 KZT. If the target falls below its lower bound of 1,000 KZT, it switches to its upper bound of 10,000 KZT.
- **CartelTwoBound** matches A when A's price rounds to 6,600 KZT. Otherwise it follows TwoBound.

The reactive rules and their default KZT parameters come from the pre-cleanup repository. Change them under `competitors` in `gym_esim.json`.

~~~sh
python run_experiments.py --algorithm ql
python run_experiments.py --algorithm sac --competitor cartel
python run_experiments.py --train-steps 5000 --eval-steps 1000 --runs 1 --output-dir experiments/esim_pilot
python run_analysis.py --input-dir experiments/esim_pilot --output-dir results/esim_pilot
python plot_results.py --input-dir experiments/esim_pilot --output-dir results/esim_pilot/figures
~~~

Use `--config` to supply a different market configuration. Repeating a command with the same output directory and settings skips completed runs and restarts incomplete runs from the beginning. Use a different output directory for changed settings. Q-learning's configured replay capacity is 16,384. Shortened runs at or below that horizon use half the training horizon as capacity, with a minimum of one. Smoke runs use capacity 16 and sample size 33 to exercise a partial minibatch. The runner reports and records the effective settings.

## Market and evaluation

By default, training uses a weighted reward: `0.7 * provider_A_revenue + 0.3 * (provider_A_revenue - provider_B_revenue)`. This gives 70% weight to revenue generation and 30% to beating the opponent. Evaluation still reports each provider's actual revenue and revenue difference in KZT. Use `--reward own_revenue` or `--reward revenue_lead` to compare the two component objectives separately. Each history row records the selected reward and each provider's actual revenue in KZT.

The market draws a fresh Poisson customer batch in each half-period. Both providers compete for each customer, who buys from A, buys from B, or declines. A reprices before the first customer batch. B keeps its previous price for that batch, then responds to A before the second batch. Each seller's price updates the reference price. Revenue sums price times sales at the price actually charged in each half-period. The two-element observation contains the reference price and the competitor price, and both algorithms scale observations using the configured price bounds.

The runner seeds Python, NumPy, and TensorFlow, and assigns separate customer seeds to training and evaluation. Corresponding Q-learning and SAC runs receive matched evaluation seeds. Evaluation uses greedy Q-learning actions and deterministic SAC actions. Updates occur only during training. Q-learning preserves its original target normalization of 1,000 KZT per reward unit. SAC divides the training reward by the configured `sac_reward_scale`, which defaults to 1,000. Histories always record unscaled KZT revenues and rewards.

Each completed run under `experiments/esim/{ql,sac}_vs_{fixed,twobound,cartel}/run_NN/` contains:

- `conf.json` with actual settings, model and customer seeds, and completion status.
- `train_history.jsonl` and `eval_history.jsonl` with one record per step, streamed to disk. They include prices, sales, non-purchases, arrivals, revenues, reward, reference price, and half-period accounting.
- `trained_model.weights.h5` for Q-learning or `trained_model.zip` for SAC. To reload the Q-learning policy, instantiate the wrapper with the recorded model settings and discrete market, then call `load_weights`. SAC checkpoints use Stable Baselines' `SAC.load`.

## Research outputs

`run_analysis.py` uses completed evaluation histories. It writes `results/individual_runs.csv` and `results/evaluation_summary.csv`. The summary includes means, sample standard deviations, medians, Student's t 95% confidence intervals, win rates, and matched SAC-minus-Q-learning differences within each competitor strategy. The within-run A-minus-B revenue difference measures competitive performance. Undefined ratios and confidence intervals with fewer than two observations are blank.

`plot_results.py` writes the following 300 DPI figures under `results/figures/`. Error bars and trajectory bands show one sample standard deviation across runs where available. Use `--price-start`, `--price-steps`, and `--smoothing-window` to change the displayed range and smoothing. Both analysis commands accept `--input-dir` and `--output-dir`. Plots for missing matchups are skipped.

| Figure files | Description |
| --- | --- |
| `revenue_comparison.png` | Mean testing revenue for both providers by matchup. |
| `<algorithm>_vs_<competitor>_revenue.png` | Both providers' testing revenue by independent run. |
| `revenue_advantage.png` | Mean percentage revenue advantage over the competitor. |
| Cumulative revenue PNGs for each matchup | Mean cumulative testing revenue for both providers. |
| `price_trajectories.png` | Evaluation prices for both providers, showing B's price in each half-period. |
| `learning_convergence.png` | Moving average of unscaled training revenue using a common window. |
| `revenue_boxplot.png` | Testing revenue distributions across runs. |
| `market_share.png` | Mean share of completed purchases for both providers. |
