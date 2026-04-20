library(tidyr)
library(dplyr)
library(readr)

args <- commandArgs(T)
godmc_02_results <- args[1]
cohort <- args[2]
hrc_ref <- args[3]

ref <- read_delim(hrc_ref)
ref$variant_index <- c(1:nrow(ref))-1

encode <- read_delim(paste0(godmc_02_results,"/",cohort,"_02/02/data.allele_codes.gz"), col_names=T)
colnames(encode)[1] <- "ID"
o <- merge(ref, encode, by.x="ID")
o$swap <- -1
o$swap[o$COUNTED != o$str_allele1] <- 1
o1 <- o[order(o$variant_index),]

missing <- encode %>% filter(!ID %in% o1$ID)
write.table(o1, file = paste0(godmc_02_results, "/", cohort, "_data_allele_flip.txt"), col=T, row=F, sep="\t", quote=F)
write.table(missing, file = paste0(godmc_02_results, "/", cohort, "_data_allele_missing_in_ref.txt"), col=T, row=F, sep="\t", quote=F)
