library(arrow)
library(dplyr)
library(tidyr)
library(purrr)
library(meffil)
library(stringr)

args <- commandArgs(T)

cpg_chunk <- args[1]
setwd(paste0("results/meta_analysis_cpg_chunk",cpg_chunk))

res <- list()
for (genetic_chunk in c(1:100)){
	res_path = paste0("meta_analysis_cpg_chunk", cpg_chunk, "_genetic_chunk", genetic_chunk, "_26cohorts/meta/snpid_output")
	files = list.files(path = res_path, full.name=T)
	o <- map_df(files, function(x) read_parquet(x) %>% mutate(filename = gsub(res_path,"",x)))
	o$cpg <- gsub(paste0("_genetic_chunk",genetic_chunk,"_26cohorts.parquet"),"",
		      gsub("/phenotype=", "", o$filename))
	o <- o %>% dplyr::select(-filename)
	res[[genetic_chunk]] <- o
}

output <- bind_rows(res)

annots_epic <- meffil.get.features("epic") %>% dplyr::select(name, chromosome, position)
annots_epic2 <- meffil.get.features("epic2") %>% dplyr::select(name, chromosome, position)
annots <- rbind(annots_epic, annots_epic2) %>% unique() %>%
        mutate(chr_cpg = gsub("chr","",chromosome), pos_cpg = as.numeric(position)) %>%
        filter(chr_cpg %in% c(1:22)) %>%
        dplyr::select(name, chr_cpg, pos_cpg)
repeatCpG <- annots %>% group_by(name) %>% tally() %>% filter(n==2) # cpgs with different positions in EPIC/EPIC2 annotation
annots <- annots %>% filter(!(name %in% repeatCpG$name))

cpglist <- unique(output$cpg)
message(paste0(length(cpglist)," CpGs are included in the summary statistics"))

process_all_cis_trans <- function(output, cpglist, annots) {
  df <- output %>% filter(cpg %in% cpglist)
  
  annots_sub <- annots %>% dplyr::select(name, chr_cpg, pos_cpg)
  df <- df %>% left_join(annots_sub, by = c("cpg" = "name"), suffix = c("", "_annot"))
  
  if ("chr_cpg_annot" %in% names(df)) {
    df$chr_cpg <- coalesce(df$chr_cpg_annot, df$chr_cpg)
    df$pos_cpg <- coalesce(df$pos_cpg_annot, df$pos_cpg)
    df <- df %>% select(-ends_with("_annot"))
  }
  
  df$SNP <- df$ID
  parts <- str_split_fixed(df$ID, "[:_]", 4)
  df$chr_snp <- parts[, 1]
  df$pos_snp <- as.numeric(parts[, 2])
  
  diff_pos <- abs(df$pos_snp - df$pos_cpg)
  same_chr <- df$chr_snp == df$chr_cpg
  
  df <- df %>% mutate(category = case_when(
        	!same_chr ~ "Inter-Trans",
        	same_chr & diff_pos > 1e7 ~ "Intra-Trans",
        	same_chr & diff_pos > 1e6 & diff_pos <= 1e7 ~ "Long-Cis",
        	same_chr & diff_pos <= 1e6 ~ "Cis",
        	TRUE ~ "Inter-Trans"))
  
  return(df)
}

meta_res_df <- process_all_cis_trans(output, cpglist, annots)
write.table(meta_res_df, paste0("cpg_chunk_", cpg_chunk, "_meta_results_100_genetic_chunks_combine_t4.42.csv"), col=T, row=F, sep=",", quote=F)

message(paste0("All summary statistics of cpg chunk ", cpg_chunk, " have been aggregated."))

