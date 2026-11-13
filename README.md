# Telemetry-source valuation and leakage-audited explainable alerts on RT-IoT2022

Reproducible study of a two-pass explainable random-forest detector for IoT security monitoring, treated as an information-integration problem: which telemetry an edge collector must extract and ship, what each source contributes, what happens when a source fails, and which alerts should go to human review.

This repository accompanies the article *Which Telemetry Must the Edge Ship? Leakage-Audited Explainable Alerts and Telemetry-Source Valuation for IoT Security Monitoring*. Every figure and table value in the article is regenerated from the files here.

## Data

RT-IoT2022 (Sharmila and Nagapadma, *Cybersecurity* 6, 2023), UCI Machine Learning Repository, dataset 942, CC BY 4.0: 123,117 bidirectional flows, 83 traffic features, nine attack classes and three normal device behaviours (`Thing_Speak`, `MQTT_Publish`, `Wipro_bulb`). The data are not redistributed here. The file used has SHA-256 `956956c09c1764584fa08acd0f6876475626bcedcd6a6b1f8c492c2e9a2089ea`. Its first column, `Unnamed: 0`, is a per-class row counter written at export; it is excluded from all models except where its effect is measured.

## Contents

| File | Purpose |
|---|---|
| `rtiot_xai.py` | Data loading, telemetry-source grouping, two-pass detector with class-balanced TreeSHAP ranking, leakage audit, coalition models and exact Shapley attribution over telemetry sources, degraded-telemetry evaluation |
| `run_rtiot.py` | Runs all experiments over five stratified 60/40 splits and writes raw and summary results (about 15 min on two CPU cores) |
| `review_routing.py` | Routes the least confident alerts to review and measures the share of errors captured |
| `make_figures.py` | Regenerates the article figures, the example alert records and the confidence bins |
| `RTIoT_Telemetry_Experiments.ipynb` | Colab notebook running everything end to end |
| `Outputs/Others/` | Raw per-split results (CSV) |
| `Outputs/Tables/` | Summary tables with 95 % confidence intervals (CSV) and example alert records (JSON) |
| `Outputs/Figures/` | Article figures (PNG, 300 dpi) |

## Reproduce

```bash
pip install -r requirements.txt
python run_rtiot.py path/to/rt-iot2022.csv
python review_routing.py path/to/rt-iot2022.csv
python make_figures.py
```

## Telemetry sources

| Source | Features |
|---|---|
| Timing | flow duration, inter-arrival times, active and idle periods, rates per second (30) |
| Volume and payload | packet counts, payload statistics, subflow and bulk statistics, down/up ratio (30) |
| Header, flags, window | header sizes, TCP flag counts, initial and last window sizes (19) |
| Protocol and service | transport protocol, application service (2) |
| Endpoint ports | originating and responding port (2) |

## Scope

Results come from one testbed capture with random stratified splits; flows of the same attack run can appear in both training and test data. Timing measurements indicate relative cost only. Review routing assumes that a reviewed alert is corrected.

## Citation

See `CITATION.cff`.

## License

MIT
