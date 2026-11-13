library(arrow)
library(dplyr)
library(tidyr)
library(purrr)

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
	o <- o %>% select(-filename)
	res[[genetic_chunk]] <- o
}

result <- bind_rows(res)
write.table(result, paste0("cpg_chunk_", cpg_chunk, "_meta_results_100_genetic_chunks_combine_t4.42.csv"), col=T, row=F, sep=",", quote=F)

message(paste0("All summary statistics of cpg chunk ", cpg_chunk, " have been aggregated."))
