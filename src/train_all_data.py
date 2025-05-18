import pandas as pd

def train_data():
    pass

def train_all():
    pass

def init():
    pass

def main():
    train_configs = {
        "prefix": ["cDNA-NA1278","dNA-NA1278", "pacbio_ENCFF450VAU", "SRR307903"],
        "bam_file": ["data/nanopore_cDNA_NA12878/NA12878-cDNA.sorted.bam",
                     "data/nanopore_dRNA_NA12878/NA12878-DirectRNA.sorted.bam",
                     "data/pacbio_ENCFF450VAU/ENCFF450VAU.sorted.bam",
                     "data/SRR307903_hisat/hisat.sorted.bam"],
        "gtf_file1": ["data/nanopore_cDNA_NA12878/stringtie.gtf", 
                      "data/nanopore_dRNA_NA12878/stringtie.gtf",
                      "data/pacbio_ENCFF450VAU/stringtie.gtf",
                      "data/SRR307903_hisat/stringtie.gtf"],
        "gtf_file2": ["data/nanopore_cDNA_NA12878/isoquant.gtf", 
                      "data/nanopore_dRNA_NA12878/isoquant.gtf",
                      "data/pacbio_ENCFF450VAU/isoquant.gtf",
                      "data/SRR307903_hisat/scallop2.gtf"],
        "tmap_file1":  ["data/nanopore_cDNA_NA12878/stringtie.stringtie.gtf.tmap",
                        "data/nanopore_dRNA_NA12878/stringtie.stringtie.gtf.tmap",
                       "data/pacbio_ENCFF450VAU/stringtie.stringtie.gtf.tmap",
                       "data/SRR307903_hisat/stringtie.stringtie.gtf.tmap"],
        "tmap_file2":  ["data/nanopore_cDNA_NA12878/isoquant.isoquant.gtf.tmap",
                        "data/nanopore_dRNA_NA12878/isoquant.isoquant.gtf.tmap",
                       "data/pacbio_ENCFF450VAU/isoquant.isoquant.gtf.tmap",
                       "data/SRR307903_hisat/scallop2.scallop2.gtf.tmap"]

    }

    t = pd.DataFrame(train_configs)
    print(t)


if __name__ == "__main__":
    main()