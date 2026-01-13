rm(list = ls())
library(limma)
library(stringr)
library(caret)
library(pheatmap)
library(RColorBrewer)
library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)


##### Caculate DMPs for PRAD #####
matrix_names<-c("tumor_PRAD.tsv","normal_PRAD.tsv", "normal_plasma.tsv","ctProstate_Resistant.tsv","ctProstate_Sensitive.tsv")
matrix_list <- lapply(martix_names,function(x){read.table(paste0(x),header = T,sep = "\t",row.names = 1)})
names(matrix_list) <- sub("\\.tsv$", "", martix_names)

# Remove methylation probes with missing values in any sample.
probs_to_keep <- Reduce(intersect, lapply(matrix_list, function(x)
  {rownames(na.omit(x))}))
matrix_list <- lapply(matrix_list ,function(x) {x[probs_to_keep,]})

beta_matrix <- do.call(cbind, matrix_list[1:3])
colnames(beta_matrix) <- c(paste0('tumor_',1:ncol(matrix_list[[1]])),paste0('normal_',1:ncol(matrix_list[[2]])),paste0('plasma_',1:ncol(matrix_list[[3]])))

group_sizes <- sapply(matrix_list[1:3], ncol)
group_names <- names(matrix_list[1:3])
groups <- factor(rep(group_names, times = group_sizes),levels = group_names)
group_mapping <- data.frame(sample_id = colnames(beta_matrix),group = groups)

# Transform the beta-value matrix into an M-value matrix using the logit function for subsequent statistical analyses.
M_matrix <- pmin(pmax(as.matrix(beta_matrix), 1e-6), 1 - 1e-6) %>%
  qlogis()

plot_methylation_density  <- function(mat, filename) {
  p <- as.data.frame(mat) %>% rownames_to_column("cpg_id") %>%     
  pivot_longer(-cpg_id, names_to = "sample_id", values_to = "value") %>%     
  merge(group_mapping, by = "sample_id") %>% 
  ggplot(aes(value, color = group)) + 
  geom_density() + 
  theme_bw() + 
  theme(legend.position = "bottom", 
  plot.title = element_text(hjust = 0.65, size = 20, face = "bold"), 
  axis.title = element_text(size = 16), axis.text = element_text(size = 14), 
  legend.text = element_text(size = 13), legend.title = element_blank(), 
  panel.border = element_rect(color = "black", fill = NA, linewidth = 1))

ggsave(filename, p, width = 8, height = 6, dpi = 300) }
plot_methylation_density(beta_matrix[1:2000,],"Methylation_beta_value.jpg")
plot_methylation_density(M_matrix[1:2000,],"Methylation_M_value.jpg")

mean_beta <- apply(beta_matrix, 1, mean)
var_beta  <- apply(beta_matrix, 1, var)
mean_M <- apply(M_matrix, 1, mean)
var_M  <- apply(M_matrix, 1, var) 
jpeg("variance.beta_vs_M.jpg", width = 16, height = 6, res = 300, units = 'in')
par(mfrow = c(1, 2),mar = c(5.1, 5.1, 4.1, 2.1))
plot(mean_beta,var_beta,main = "",xlab = "Mean Beta-value",ylab = "Variance Beta-value",
  pch = 16,cex = 0.4, col = "#8B000033" , cex.lab = 1.4,cex.axis = 1.2) 
grid()
plot(mean_M, var_M,main = "",xlab = "Mean M-value",ylab = "VarianceM-value",
  pch = 16,cex = 0.4,col = "#8B000033", cex.lab = 1.4,cex.axis = 1.2)
grid()
dev.off()

# Identify tumor-specific DMPs
design <- model.matrix(~0+groups)
colnames(design) <- levels(groups)
fit <- lmFit(M_matrix, design)

contMatrix <- makeContrasts(tumor_PRAD - normal_PRAD, normal_PRAD -normal_plasma, levels=design)
fit2 <- contrasts.fit(fit, contMatrix)
fit2 <- eBayes(fit2)

tumor_results <- topTable(fit2, number = Inf, adjust.method = "fdr",coef = 1)
tissue_results <- topTable(fit2, number = Inf, adjust.method = "fdr",coef = 2)

tissue_not_sig_probes <- tissue_results %>% filter(adj.P.Val >= 0.05 | abs(logFC) < 0.01) %>%  rownames()
tumor_sig <- tumor_results %>% rownames_to_column("probe_id") %>%     
  filter( adj.P.Val < 0.01, abs(logFC) > 0.2, probe_id %in% tissue_not_sig_probes ) %>% 
  arrange(desc(abs(logFC))) %>% slice_head(n = 500) %>% 
  pull(probe_id)

# Generate a heatmap of the top 500 tumor-specific sites
annotation_col <- data.frame(Group = factor(rep(group_names, times = group_sizes), levels = group_names))
rownames(annotation_col) <- colnames(beta_matrix)

pheatmap( 
  as.matrix(beta_matrix[tumor_sig,]), 
  annotation_col = annotation_col, 
  scale = "row", 
  cluster_rows = TRUE, 
  cluster_cols = FALSE,   
  treeheight_row = 0,
  color = colorRampPalette(rev(brewer.pal(7, "RdYlBu")))(100), 
  show_rownames = FALSE, 
  show_colnames = FALSE,
  filename = "PRAD_top500.jpg")

# Save the training and test sets for Oncoder
train_matrix <- beta_matrix[tumor_sig,which(grepl("tumor",colnames(beta_matrix))|grepl("plasma",colnames(beta_matrix)))]
ctProstate_Resistant<- matrix_list$ctProstate_Resistant[tumor_sig,]
ctProstate_Sensitive <- matrix_list$ctProstate_Sensitive[tumor_sig,]

write.table(ctProstate_Sensitive, file = "test_ctPRAD_Sensitive.tsv", sep = "\t", quote = FALSE)
write.table(ctProstate_Resistant, file = "test_ctPRAD_Resistant.tsv", sep = "\t",quote = FALSE)
write.table(train_matrix, file = "train_matrix_PRAD.tsv",sep = "\t",quote = FALSE)
