library(tidyr)
library(dplyr)
library(readr)
library(purrr)

setwd("./Positive_Control/metal_input")
files <- list.files(path=".", pattern="meQTL")

df <- lapply(files, function(x) read_delim(x,col_names=F) %>% mutate(cohort=x))
df1 <- bind_rows(df)
colnames(df1) <- c("chr","pos","ID","REF","ALT","A1","TEST","OBS_CT","BETA","SE","T","P","ERRCODE","A2","cohort")
df1$cohort <- gsub("_metal_input.txt","",gsub("meQTL_","",df1$cohort))

df1 <- df1 %>% select(BETA, SE, cohort)
df1$lower <- df1$BETA - 1.96*df1$SE
df1$upper <- df1$BETA + 1.96*df1$SE
df1$cohort <- factor(df1$cohort, levels=rev(df1$cohort))

library(ggplot2)
p <- ggplot(df1, aes(x = BETA, y = cohort)) +
	geom_errorbarh(aes(xmin = lower, xmax = upper), height = 0.2, color = "black", size = 0.8) +
	geom_point(aes(size = 1/SE), color = "darkblue", shape = 15) +
	geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
	theme_minimal() +
  	labs(
    	title = "cg07959070 (22:50053871)",
    	x = "Effect Size (Beta) & 95% CI",
    	y = ""
  	) +
  	theme(legend.position = "none")

jpeg("Forest_plot_meQTL_cg07959070.jpeg")
p
dev.off()
