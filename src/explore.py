import pandas as pd
tss_labeled = pd.read_csv('train_output/cDNA-NA12878_stringtie/features/cDNA-NA12878_stringtie_tss_features.tsv')
sc_labeled = pd.read_csv('train_output/cDNA-NA12878_stringtie/features/cDNA-NA12878_stringtie_tss_softclip_labeled.tsv')

sc_labeled = sc_labeled[sc_labeled['clip_type'] == 'start']
grouped_sc = sc_labeled.groupby(['chrom', 'position', 'strand'])

okcnt = 0
badcnt = 0
for site, site_data in grouped_sc:
    tss_row = tss_labeled[(tss_labeled['chrom'] == site[0]) & (tss_labeled['position'] == site[1]) & (tss_labeled['strand'] == site[2])]
    if len(tss_row) > 1:
        print(site)
        print(tss_row)
        continue
    tss_row = tss_row.iloc[0]
    if len(site_data) <= 200:
        if abs(tss_row['start_soft_clip_count'] - len(site_data)) >= 1:
            badcnt += 1
            continue
        else:
            okcnt += 1

print(okcnt)
print(badcnt)