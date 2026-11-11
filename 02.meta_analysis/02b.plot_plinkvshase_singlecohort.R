library(tidyr)
library(readr)
library(dplyr)
library(arrow)
library(purrr)
library(ggplot2)
library(stringr)

args <- commandArgs(T)

results03 <- args[1]
hase_path <- args[2]
cohort <- args[3]
ref_hrc <- args[4]
metal_path <- args[5]

plink <- read_delim(paste0(results03, "/", cohort, "_03/03/positive_control_transformed_cg07959070.PHENO1.glm.linear.gz")) %>% filter(TEST=="ADD")
plink <- plink %>%
  mutate(across(where(is.character), ~ str_replace_all(., "^X$", "23")))

plink$A2 <- plink$ALT
plink$A2[plink$A1==plink$ALT] <- plink$REF[plink$A1==plink$ALT]

plink %>% filter(A1==A2) %>% nrow() # should be 0
write.table(plink, paste0(metal_path, "/", cohort,"_metal_input.txt"), col=T, row=F, sep="\t", quote=F)

hase_output <- data.frame()
for (chr in c(2:22)){
hase_files <- list.files(path=paste0(hase_path, "/positive_control_chr", chr, "/cohort/cohort=", cohort,"/phenotype=cg07959070/"))
	setwd(paste0(hase_path, "/positive_control_chr", chr, "/cohort/cohort=", cohort,"/phenotype=cg07959070/"))
	hase_combined <- map_df(hase_files, read_feather)
	hase_output <- rbind(hase_output, hase_combined)
}

ref <- read_delim(ref_hrc)
ref$variant_index <- c(1:nrow(ref))-1
df <- merge(hase_output, ref, by.x="variant_index") %>%
    mutate(hase_beta = beta, hase_se = standard_error) %>%
    select(ID, hase_beta, hase_se, str_allele1, str_allele2) %>% na.omit()

df1 <- merge(plink, df, by.x="ID")
df1$BETA[df1$A1==df1$str_allele2] <- -1*df1$BETA[df1$A1==df1$str_allele2]
colnames(df1)[9] <- "PLINK_Beta"
colnames(df1)[15] <- "HASE_Beta"

df_same <- df1 %>% filter(OBS_CT== max(df1$OBS_CT))
message(paste0("same sample size is:", nrow(df_same)))

jpeg(paste0("/projects/b35cx/XiaopuZ/GoDMC2_mqtl_meta_analysis/02.meta_analysis/phenotype=cg07959070_", cohort, "_N=", max(df1$OBS_CT), "_PLINKvsHASE_BETA.jpeg"),quality=75)
ggplot(df_same, aes(x = PLINK_Beta, y = HASE_Beta)) + geom_point()
dev.off()

jpeg(paste0("/projects/b35cx/XiaopuZ/GoDMC2_mqtl_meta_analysis/02.meta_analysis/phenotype=cg07959070_", cohort, "_N=", max(df1$OBS_CT), "_PLINKvsHASE_SE.jpeg"),quality=75)
ggplot(df_same, aes(x = SE, y = hase_se)) + geom_point()
dev.off()

df_diff <- df1 %>% filter(OBS_CT != max(df1$OBS_CT))
message(paste0("diff sample size is:", nrow(df_diff)))

jpeg(paste0("/projects/b35cx/XiaopuZ/GoDMC2_mqtl_meta_analysis/02.meta_analysis/phenotype=cg07959070_", cohort, "_N<", max(df1$OBS_CT), "_PLINKvsHASE_BETA.jpeg"),quality=75)
ggplot(df_diff, aes(x = PLINK_Beta, y = HASE_Beta)) + geom_point()
dev.off()

jpeg(paste0("/projects/b35cx/XiaopuZ/GoDMC2_mqtl_meta_analysis/02.meta_analysis/phenotype=cg07959070_", cohort, "_N<", max(df1$OBS_CT), "_PLINKvsHASE_SE.jpeg"),quality=75)
ggplot(df_diff, aes(x = SE, y = hase_se)) + geom_point()
dev.off()
