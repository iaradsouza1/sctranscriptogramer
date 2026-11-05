#!/usr/bin/env Rscript
# Maintainer one-off: export the protein orderings shipped in the R `transcriptogramer`
# package to plain CSVs (protein, position), which build_ordering_assets.py then converts
# to the parquet files bundled in the wheel.
#
# The ordering is the ONLY asset reused from the R package. Positions are 0-based, matching
# the R package convention (transcriptogramer:::check_ordering assigns 0..N-1).
#
# Usage:
#   Rscript export_orderings_from_r.R <r_package_data_dir> <out_csv_dir>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: export_orderings_from_r.R <r_package_data_dir> <out_csv_dir>")
}
data_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

orderings <- c(
  "Hs700", "Hs800", "Hs900",
  "Mm700", "Mm800", "Mm900",
  "Rn700", "Rn800", "Rn900",
  "Sc700", "Sc800", "Sc900"
)

for (name in orderings) {
  path <- file.path(data_dir, paste0(name, ".RData"))
  if (!file.exists(path)) {
    message("skip (missing): ", path)
    next
  }
  env <- new.env()
  load(path, envir = env)
  obj <- get(name, envir = env)
  if (!is.character(obj)) {
    stop(sprintf("ordering %s is not a character vector (got %s)", name, class(obj)[1]))
  }
  df <- data.frame(
    protein = obj,
    position = seq_len(length(obj)) - 1L, # 0-based
    stringsAsFactors = FALSE
  )
  out <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(df, out, row.names = FALSE)
  message(sprintf("wrote %s (%d proteins)", out, nrow(df)))
}
