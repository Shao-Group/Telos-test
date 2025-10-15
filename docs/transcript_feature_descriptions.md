### Transcript-level feature descriptions (from `src/train_model.py`)

This document describes transcript-level features engineered in `src/train_model.py`. Many features are derived from exon-level inputs: `first_exon_length`, `last_exon_length`, `max_exon_length`, `min_exon_length`, `mean_exon_length`, `total_exon_length`, and `exon_count`, as well as global signals like `coverage`, `total_reads`, and boundary confidences (`probability_tss` and `probability_tes`, used here as `tss_confidence` and `tes_confidence`). Some features are only present when the required input columns exist.

| Feature | Description |
| --- | --- |
| transcript_length | Absolute distance between `tes_pos` and `tss_pos`. |
| log_transcript_length | `log1p(transcript_length)`; log-scaled length to reduce skew. |
| tss_confidence | Confidence score for the TSS (`probability_tss`, default 0.5 if missing). |
| tes_confidence | Confidence score for the TES (`probability_tes`, default 0.5 if missing). |
| min_confidence | `min(tss_confidence, tes_confidence)`; weakest boundary confidence. |
| confidence_product | `tss_confidence * tes_confidence`; joint boundary confidence. |
| exon_density | `exon_count / transcript_length` (length clipped ≥ 1); exons per bp. |
| confidence_exon_interaction | `confidence_product * exon_count`; boundaries × exon count. |
| coverage_per_exon | `coverage / exon_count` (count clipped ≥ 1); coverage normalized by exon number. |
| avg_exon_length | `total_exon_length / exon_count` (count clipped ≥ 1); mean exon size. |
| exon_length_ratio | `max_exon_length / min_exon_length` (min clipped ≥ 1); exon size heterogeneity. |
| coverage_length_ratio | `coverage / transcript_length` (length clipped ≥ 1); coverage per bp. |
| confidence_coverage_interaction | `confidence_product * coverage`; boundaries × expression. |
| confidence_sum | `tss_confidence + tes_confidence`; total boundary support. |
| confidence_diff | `abs(tss_confidence - tes_confidence)`; boundary asymmetry. |
| terminal_exon_ratio | `first_exon_length / last_exon_length` (denominator clipped ≥ 1); 5′ vs 3′ terminal exon balance. |
| exon_efficiency | `mean_exon_length / transcript_length` (length clipped ≥ 1); average exon size relative to transcript. |
| log_coverage | `log1p(coverage)` (only if `coverage` exists). |
| log_total_reads | `log1p(total_reads)` (only if `total_reads` exists). |
| exon_length_entropy | `log1p((max_exon_length - min_exon_length) / mean_exon_length)` (mean clipped ≥ 1); proxy for exon length dispersion. |
| exon_length_std | `sqrt(exon_length_variance)` if available; 0 otherwise. |
| exon_length_skewness | `(max_exon_length - mean_exon_length) / exon_length_std` (std clipped ≥ 1); asymmetry of exon length distribution. |

Notes:
- “Clipped ≥ 1” means denominators are lower-bounded at 1 to avoid division by zero.
- Features that depend on optional columns appear only when those columns are present in the input dataframe.

### Motivations for transcript-level features

- **transcript_length / log_transcript_length**: Captures isoform span; log scaling stabilizes heavy-tailed distributions.
- **tss_confidence, tes_confidence, min_confidence, confidence_sum, confidence_product, confidence_diff**: Boundary quality signals; product encodes joint support, min captures weakest link, sum aggregates evidence, diff highlights imbalance between 5′ and 3′ boundaries.
- **exon_density**: Normalizes exon count by length; compact, exon-dense transcripts may exhibit distinct expression and boundary patterns.
- **confidence_exon_interaction**: Tests whether high-confidence boundaries co-occur with complex exon structures.
- **coverage_per_exon, coverage_length_ratio, log_coverage, log_total_reads**: Expression-normalized indicators that reduce bias from depth and transcript size.
- **avg_exon_length, exon_length_ratio**: Summarize exon size and heterogeneity; extreme ratios may indicate retained introns or alternative splicing.
- **terminal_exon_ratio**: Contrasts 5′ and 3′ terminal exon sizes; informative for transcript boundary fidelity and processing.
- **exon_efficiency**: Fraction of transcript length captured by the average exon; lower values suggest many small exons or long introns.
- **exon_length_entropy (proxy), exon_length_std, exon_length_skewness**: Dispersion and asymmetry of exon lengths can reflect alternative splicing complexity and sequencing/alignment artifacts.




### LaTeX table

Use the following LaTeX snippet to include the transcript-level feature description table in manuscripts.

```latex
\begin{table}[t]
  \centering
  \small
  \caption{Transcript-level feature descriptions engineered in src/train_model.py}
  \label{tab:transcript_features}
  \begin{tabular}{p{0.28\linewidth} p{0.66\linewidth}}
    \toprule
    \textbf{Feature} & \textbf{Description} \\
    \midrule
    transcript\_length & Absolute distance between $tes\_pos$ and $tss\_pos$. \\
    log\_transcript\_length & $\log(1+\text{transcript\_length})$; log-scaled length to reduce skew. \\
    tss\_confidence & Confidence score for the TSS ($probability\_tss$, default 0.5 if missing). \\
    tes\_confidence & Confidence score for the TES ($probability\_tes$, default 0.5 if missing). \\
    min\_confidence & $\min(\text{tss\_confidence},\ \text{tes\_confidence})$; weakest boundary confidence. \\
    confidence\_product & $\text{tss\_confidence} \times \text{tes\_confidence}$; joint boundary confidence. \\
    exon\_density & $\text{exon\_count} / \text{transcript\_length}$ (length clipped $\ge 1$); exons per bp. \\
    confidence\_exon\_interaction & $\text{confidence\_product} \times \text{exon\_count}$; boundaries × exon count. \\
    coverage\_per\_exon & $\text{coverage} / \text{exon\_count}$ (count clipped $\ge 1$); coverage normalized by exon number. \\
    avg\_exon\_length & $\text{total\_exon\_length} / \text{exon\_count}$ (count clipped $\ge 1$); mean exon size. \\
    exon\_length\_ratio & $\text{max\_exon\_length} / \text{min\_exon\_length}$ (min clipped $\ge 1$); exon size heterogeneity. \\
    coverage\_length\_ratio & $\text{coverage} / \text{transcript\_length}$ (length clipped $\ge 1$); coverage per bp. \\
    confidence\_coverage\_interaction & $\text{confidence\_product} \times \text{coverage}$; boundaries × expression. \\
    confidence\_sum & $\text{tss\_confidence} + \text{tes\_confidence}$; total boundary support. \\
    confidence\_diff & $|\text{tss\_confidence} - \text{tes\_confidence}|$; boundary asymmetry. \\
    terminal\_exon\_ratio & $\text{first\_exon\_length} / \text{last\_exon\_length}$ (denominator clipped $\ge 1$); 5′ vs 3′ terminal exon balance. \\
    exon\_efficiency & $\text{mean\_exon\_length} / \text{transcript\_length}$ (length clipped $\ge 1$); average exon size relative to transcript. \\
    log\_coverage & $\log(1+\text{coverage})$ (only if coverage exists). \\
    log\_total\_reads & $\log(1+\text{total\_reads})$ (only if total\_reads exists). \\
    exon\_length\_entropy & $\log\!\left(1+\frac{\text{max\_exon\_length}-\text{min\_exon\_length}}{\max(1,\ \text{mean\_exon\_length})}\right)$; proxy for exon length dispersion. \\
    exon\_length\_std & $\sqrt{\text{exon\_length\_variance}}$ if available; 0 otherwise. \\
    exon\_length\_skewness & $\frac{\text{max\_exon\_length}-\text{mean\_exon\_length}}{\max(1,\ \text{exon\_length\_std})}$; asymmetry of exon length distribution. \\
    \bottomrule
  \end{tabular}
\end{table}
```
