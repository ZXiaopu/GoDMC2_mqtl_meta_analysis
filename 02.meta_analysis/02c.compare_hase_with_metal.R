library(arrow)
library(tidyr)
library(dplyr)
library(readr)
library(purrr)
library(ggplot2)

load("./Positive_Control/metal_input/Metal_result_glaku_pisa_filtered_samplesize.RData")

hase_files <- list.files(path="./Positive_Control/results/positive_control_2cohorts/phenotype=cg07959070/",pattern="csv")
setwd("./Positive_Control/results/positive_control_2cohorts/phenotype=cg07959070/")
hase_combined <- map_df(hase_files, read_delim)

hase_combined1 <- hase_combined %>% mutate(MarkerName=ID) %>% filter(MarkerName %in% meta_filtered$MarkerName)

df <- merge(hase_combined1, meta_filtered, by.x="MarkerName")
df$Allele1 <- toupper(df$Allele1)
df$Allele2 <- toupper(df$Allele2)
df$beta_metal <- df$Effect
df$beta_metal[df$Allele1 != df$effect_allele] <- -1*df$beta_metal[df$Allele1 != df$effect_allele]
df$beta_hase <- df$beta

jpeg(paste0("meta_result_pisa_glaku_METALvsHASE_SE.jpeg"),quality=75)
ggplot(df, aes(x = standard_error, y = StdErr)) + geom_point() + xlab("HASE") + ylab("METAL")
dev.off()

jpeg(paste0("meta_result_pisa_glaku_METALvsHASE_beta.jpeg"),quality=75)
ggplot(df, aes(x = beta_hase, y = beta_metal)) + geom_point() + xlab("HASE") + ylab("METAL")
dev.off()
