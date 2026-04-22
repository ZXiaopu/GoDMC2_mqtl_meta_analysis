library(data.table)
library(dplyr)

args <- commandArgs(T)
data_dir <- args[1]
hrc_ref <- args[2]
out <- args[3]

files <- list.files(data_dir, full.names = FALSE)

is_folder_name <- grepl("_02$", files)
is_file_name   <- grepl("\\.aes$", files)

folder_count <- sum(is_folder_name)
file_count   <- sum(is_file_name)

message("================================================")
message(paste("Total items recorded: ", length(files)))
message(paste("Folders (ending in _02):  ", folder_count))
message(paste("Files (ending in .aes):   ", file_count))
message("Please confirm the number is correct before proceeding.")
message("================================================")

cohort_list = files[is_folder_name]
cohort_name_list = gsub("_02", "", cohort_list)

ref = fread(hrc_ref)
# nrow(ref)
# 16430621

summary_table = data.frame(Cohort = character(), Total_SNPs = integer(), SNPs_not_in_ref = integer(), Ref_SNPs_not_in_cohort = integer(), stringsAsFactors = FALSE)

# find corresponding cohort data.allele_codes.gz files within the cohort folders
for (cohort in cohort_list) {
	cohort_allele = fread(paste0(data_dir, "/" ,cohort, "/02/data.allele_codes.gz"))
	cohort_name = gsub("_02", "", cohort)
	message(paste("Processing cohort:", cohort_name))
	# print(head(cohort_allele))
	total_snps = nrow(cohort_allele)
	snps_not_in_ref = sum(!cohort_allele$SNP %in% ref$ID)
	ref_snps_not_in_cohort = sum(!ref$ID %in% cohort_allele$SNP)
	summary_table = rbind(summary_table, data.frame(Cohort = cohort_name, Total_SNPs = total_snps, SNPs_not_in_ref = snps_not_in_ref, Ref_SNPs_not_in_cohort = ref_snps_not_in_cohort))
}

write.table(summary_table, file = paste0(out,"/missing_snp_summary.txt", sep = "\t", row.names = FALSE, quote = FALSE)
