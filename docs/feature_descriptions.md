### Feature descriptions (from `src/extract_features.py`)

Note on coverage-based windows: “Adjacent windows” refer to consecutive, non-overlapping bins of fixed width centered around the site. In code, coverage is computed in 10 bp windows stepped every 10 bp across a ±100 bp range (by default). Gradients are the difference in average coverage between two neighboring 10 bp windows; sharpness normalizes this change by the local average coverage.

| Feature | Description |
| --- | --- |
| read_start_density | Number of read starts within the density window around the site. |
| read_end_density | Number of read ends within the density window around the site. |
| start_soft_clip_mean | Mean length of soft clips at read starts near the site. |
| end_soft_clip_mean | Mean length of soft clips at read ends near the site. |
| start_soft_clip_max | Maximum soft-clip length at read starts near the site. |
| end_soft_clip_max | Maximum soft-clip length at read ends near the site. |
| start_soft_clip_median | Median soft-clip length at read starts near the site. |
| end_soft_clip_median | Median soft-clip length at read ends near the site. |
| start_soft_clip_count | Count of reads with a soft clip at the start near the site. |
| end_soft_clip_count | Count of reads with a soft clip at the end near the site. |
| mean_mapq | Mean mapping quality of reads overlapping the site. |
| std_mapq | Standard deviation of mapping quality of reads overlapping the site. |
| strand_ratio | Ratio of reads on the site’s strand to reads on the opposite strand. |
| coverage_before | Base-level coverage in the upstream window relative to transcript direction. |
| coverage_after | Base-level coverage in the downstream window relative to transcript direction. |
| delta_coverage | Difference between downstream and upstream coverage (after − before). |
| nearest_splice_dist | Distance (bp) to the nearest spliced region (CIGAR N operation). |
| softclip_bias | Fraction of reads with a soft clip starting within ±5 bp of the site. |
| start_entropy | Shannon entropy of the distribution of read start positions near the site. |
| end_entropy | Shannon entropy of the distribution of read end positions near the site. |
| coverage_gradient_sharpness | Max coverage change between adjacent windows, normalized by local mean coverage. |
| max_coverage_gradient | Maximum absolute difference in coverage between adjacent windows. |
| local_avg_coverage | Mean coverage across the analyzed windows around the site. |
| avg_clip_length | Mean length of all soft-clipped sequences collected near the site. |
| coverage_efficiency | Total reads overlapping the region divided by (coverage_before + coverage_after). |
| coverage_log_ratio | log2(coverage_after / coverage_before), stabilized to avoid division by zero. |
| coverage_ratio | coverage_after / coverage_before, stabilized to avoid division by zero. |
| five_prime_degradation_score | Negative slope of read-start density across windows. |
| gc_content | Fraction of G and C bases in combined soft-clipped sequences. |
| k3_at_rich_kmers | Count of 3-mers with GC content < 0.4 in soft-clipped sequences. |
| k3_balanced_kmers | Count of 3-mers with 0.4 ≤ GC content ≤ 0.6. |
| k3_gc_kmers_ratio | Fraction of 3-mers that are GC-rich among all observed 3-mers. |
| k3_gc_rich_kmers | Count of 3-mers with GC content > 0.6. |
| k3_kmer_diversity | Number of unique 3-mers divided by total 3-mers. |
| k3_kmer_entropy | Shannon entropy of the 3-mer frequency distribution. |
| k3_most_frequent_kmer_ratio | Frequency of the most common 3-mer divided by total 3-mers. |
| k3_purine_rich_kmers | Count of 3-mers with purine (A/G) content > 0.6. |
| k3_repeat_kmers | Count of homopolymer 3-mers (e.g., AAA, TTT). |
| k3_unique_kmers | Number of unique 3-mers observed. |
| kmer_AAA, kmer_AAT, …, kmer_TTT | Counts of specific biologically relevant 3-mers in soft-clipped sequences. |
| long_to_short_clip_ratio | Number of soft clips > 20 nt divided by those ≤ 10 nt. |
| max_clip_length | Maximum length among all soft-clipped sequences near the site. |
| max_polyA / max_polyC / max_polyG / max_polyT | Longest homopolymer run of each base in soft-clipped sequences. |
| normalized_delta_coverage | (coverage_after − coverage_before) / (coverage_after + coverage_before). |
| normalized_read_density | Number of read starts divided by coverage_after (coverage-normalized density). |
| num_clips | Total number of soft-clipped sequences collected. |
| purine_ratio | Fraction of purine bases (A or G) in combined soft-clipped sequences. |
| read_start_clustering | Fraction of read starts within 5 bp of the site (clustering measure). |
| read_start_coefficient_variation | Variance of read starts divided by their mean position (stabilized). |
| read_start_variance | Variance of read start positions near the site. |
| softclip_length_skewness | Skewness of the distribution of soft-clip lengths. |
| softclip_length_variance | Variance of soft-clip lengths. |
| softclip_sparsity | Total number of soft clips divided by total overlapping reads. |
| upstream_downstream_ratio | Read starts upstream of the site divided by starts downstream. |
| weighted_read_start_density | Sum of 1/distance for read starts relative to the site. |

### Motivations for features

- **read_start_density**: Start pileups indicate potential TSS or cleavage points.
- **read_end_density**: End pileups suggest TES or truncation points.
- **start_soft_clip_mean**: Consistent start clip lengths can mark alignment breakpoints at true boundaries.
- **end_soft_clip_mean**: End soft-clips often accumulate at TES or processing events.
- **start_soft_clip_max**: Long start clips may indicate strong boundary evidence or artifacts.
- **end_soft_clip_max**: Captures extreme events consistent with cleavage/polyadenylation boundaries.
- **start_soft_clip_median**: Robust to outliers; summarizes start clip lengths.
- **end_soft_clip_median**: Robust summary of end clip lengths for boundary detection.
- **start_soft_clip_count**: Frequency of start clipping reflects interruptions at candidate TSS.
- **end_soft_clip_count**: Frequency of end clipping reflects interruptions at candidate TES.
- **mean_mapq**: High mapping quality supports true events; low suggests repeats/artifacts.
- **std_mapq**: Heterogeneous quality may indicate alignment ambiguity or mixed origins.
- **strand_ratio**: True sites are strand-consistent; helps filter antisense noise.
- **coverage_before**: Provides upstream expression context when assessing boundaries.
- **coverage_after**: Complements upstream to detect step changes at sites.
- **delta_coverage**: Captures step-like coverage changes at TSS/TES.
- **nearest_splice_dist**: True starts/ends typically avoid immediate proximity to splice junctions.
- **softclip_bias**: Enrichment of soft clips directly at the coordinate signals a boundary.
- **start_entropy**: Low entropy indicates concentrated starts typical of TSS.
- **end_entropy**: Low entropy suggests sharp TES; high entropy indicates diffuse ends.
- **coverage_gradient_sharpness**: Normalized steepness detects abrupt transitions across windows.
- **max_coverage_gradient**: Measures steepest local coverage change irrespective of scale.
- **local_avg_coverage**: Baseline expression for interpreting gradients and ratios.
- **avg_clip_length**: Longer clips can imply strong boundaries or trimming.
- **coverage_efficiency**: Normalizes read counts by coverage to reflect evidence efficiency.
- **coverage_log_ratio**: Symmetric fold-change for boundary directionality.
- **coverage_ratio**: Simple fold-change of downstream vs upstream coverage.
- **five_prime_degradation_score**: Models dRNA degradation to separate decay from genuine TSS.
- **gc_content**: GC bias relates to alignment stability and biological context.
- **k3_at_rich_kmers**: AT-richness may indicate polyA-adjacent or low-complexity contexts.
- **k3_balanced_kmers**: Marks typical background composition near boundaries.
- **k3_gc_kmers_ratio**: GC enrichment can reflect structured regions impacting alignment ends.
- **k3_gc_rich_kmers**: Tracks GC-heavy sequence contexts.
- **k3_kmer_diversity**: Higher diversity suggests complex sequence; low indicates repeats.
- **k3_kmer_entropy**: Quantifies uniformity; low entropy signals motif dominance.
- **k3_most_frequent_kmer_ratio**: Highlights dominance (e.g., polyA-related motifs).
- **k3_purine_rich_kmers**: Purine enrichment can affect structure and alignment.
- **k3_repeat_kmers**: Repeats (e.g., AAA) common at TES can cause soft clips.
- **k3_unique_kmers**: Complements entropy as a diversity measure.
- **kmer_AAA … kmer_TTT**: Motifs linked to start/termination signals and polyA contexts.
- **long_to_short_clip_ratio**: Emphasizes prevalence of long clips at sharp boundaries.
- **max_clip_length**: Captures extreme clipping events at boundaries.
- **max_polyA/max_polyC/max_polyG/max_polyT**: PolyN runs (polyA) are TES hallmarks.
- **normalized_delta_coverage**: Scale-invariant boundary strength measure.
- **normalized_read_density**: Adjusts start density by local coverage.
- **num_clips**: Overall evidence count for alignment interruptions.
- **purine_ratio**: Composition bias linked to structure and sequencing behavior.
- **read_start_clustering**: High clustering is characteristic of precise boundaries.
- **read_start_coefficient_variation**: Normalized dispersion complements raw variance.
- **read_start_variance**: Low variance indicates tightly localized start points.
- **softclip_length_skewness**: Asymmetry indicates enrichment of long/short clips.
- **softclip_length_variance**: Dispersion helps distinguish structured vs noisy regions.
- **softclip_sparsity**: Normalizes clip counts by depth to mitigate coverage bias.
- **upstream_downstream_ratio**: Directional imbalance indicates TSS/TES patterns.
- **weighted_read_start_density**: Rewards starts closer to the coordinate, emphasizing precision.


