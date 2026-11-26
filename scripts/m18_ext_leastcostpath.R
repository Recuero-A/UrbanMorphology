# M18 external baseline: "leastcostpath (Lewis) routed between seeded anchors; merge by us".
#
# leastcostpath (J. Lewis, CRAN, GPL-3) builds a slope cost surface on the site DEM and routes least-cost paths
# between every ordered pair of anchors (create_FETE_lcps). The anchors, the DEM crop and the water raster are
# written by scripts/m18_external.py; the merge of the paths into one planar street graph is done there (shapely).
#
#   Rscript scripts/m18_ext_leastcostpath.R <dem.tif> <water.tif> <anchors.csv> <out.csv> \
#           [cost_function] [crit_slope] [max_slope] [neighbours]
#   Rscript scripts/m18_ext_leastcostpath.R --example <out.csv>     # the package's create_FETE_lcps example
#
# dem.tif: the benchmark's 4 m GLO-30 grid cropped to the city limits, NA outside the limits (the package drops NA
# cells from the adjacency, so the limits bound the routing). water.tif: 1 = mapped water (barrier, conductance set
# to 0 with update_values). out.csv: one row per path vertex (path, origin, destination, x, y) in the DEM CRS.
suppressPackageStartupMessages({
  library(terra)
  library(sf)
  library(leastcostpath)
})

args <- commandArgs(trailingOnly = TRUE)
versions <- sprintf("R %s; leastcostpath %s; terra %s; sf %s; igraph %s; Matrix %s",
                    getRversion(), packageVersion("leastcostpath"), packageVersion("terra"), packageVersion("sf"),
                    packageVersion("igraph"), packageVersion("Matrix"))
cat("versions:", versions, "\n")

write_paths <- function(lcps, out) {
  if (is.null(lcps) || nrow(lcps) == 0) {
    write.csv(data.frame(path = integer(0), origin = integer(0), destination = integer(0), x = numeric(0),
                         y = numeric(0)), out, row.names = FALSE)
    return(invisible(0))
  }
  xy <- sf::st_coordinates(lcps)
  k <- xy[, "L1"]
  df <- data.frame(path = k, origin = lcps$origin_ID[k], destination = lcps$destination_ID[k], x = xy[, "X"],
                   y = xy[, "Y"])
  write.csv(df, out, row.names = FALSE)
  invisible(nrow(lcps))
}

if (length(args) >= 1 && args[1] == "--example") {
  # verbatim example of ?create_FETE_lcps (leastcostpath 2.0.13)
  r <- terra::rast(system.file("extdata/SICILY_1000m.tif", package = "leastcostpath"))
  slope_cs <- create_slope_cs(x = r, cost_function = "tobler", neighbours = 4)
  locs <- sf::st_sf(geometry = sf::st_sfc(
    sf::st_point(c(839769, 4199443)),
    sf::st_point(c(1038608, 4100024)),
    sf::st_point(c(907695, 4145478)),
    sf::st_point(c(907695, 4145478)),
    crs = terra::crs(r)))
  lcps <- create_FETE_lcps(x = slope_cs, locations = locs)
  len <- as.numeric(sf::st_length(lcps))
  cat("example: raster", nrow(r), "x", ncol(r), "res", terra::res(r)[1], "; paths", nrow(lcps),
      "; origin-destination", paste(lcps$origin_ID, lcps$destination_ID, sep = "->", collapse = " "), "\n")
  cat("example: path lengths (km)", paste(round(len / 1000, 1), collapse = " "), "\n")
  write_paths(lcps, args[2])
  quit(status = 0)
}

dem_f <- args[1]; water_f <- args[2]; anchors_f <- args[3]; out_f <- args[4]
cost_function <- if (length(args) >= 5) args[5] else "wheeled transport"
crit_slope <- if (length(args) >= 6) as.numeric(args[6]) else 12
max_slope <- if (length(args) >= 7) as.numeric(args[7]) else 20
neighbours <- if (length(args) >= 8) as.integer(args[8]) else 16L

t0 <- Sys.time()
dem <- terra::rast(dem_f)
cat("dem:", nrow(dem), "x", ncol(dem), "cells,", sum(!is.na(terra::values(dem))), "inside the limits; res",
    terra::res(dem)[1], "m\n")
cs <- create_slope_cs(x = dem, cost_function = cost_function, neighbours = neighbours, crit_slope = crit_slope,
                      max_slope = max_slope)
cat("conductance: cost function", cost_function, "; crit_slope", crit_slope, "%; max_slope", max_slope,
    "%; neighbours", neighbours, "; non-zero links", Matrix::nnzero(cs$conductanceMatrix), "\n")

# water barrier: conductance into water cells set to 0 (package update_values)
water <- terra::rast(water_f)
wpoly <- terra::as.polygons(water, values = TRUE)
wpoly <- wpoly[wpoly[[1]][, 1] == 1, ]
if (nrow(wpoly) > 0) {
  before <- Matrix::nnzero(cs$conductanceMatrix)
  cs <- update_values(x = cs, sf = sf::st_as_sf(wpoly), FUN = function(j) j * 0)
  cat("water barrier: links removed", before - Matrix::nnzero(cs$conductanceMatrix), "\n")
} else {
  cat("water barrier: no water cell in the crop\n")
}

anc <- read.csv(anchors_f)
locs <- sf::st_as_sf(anc, coords = c("x", "y"), crs = terra::crs(dem))
# create_FETE_lcps stops if any location cannot be entered from a neighbour; such anchors are dropped (logged)
cs_rast <- terra::rast(nrow = cs$nrow, ncol = cs$ncol, xmin = cs$extent[1], xmax = cs$extent[2],
                       ymin = cs$extent[3], ymax = cs$extent[4], crs = cs$crs)
cells <- terra::cellFromXY(cs_rast, sf::st_coordinates(locs))
ok <- vapply(cells, function(j) !is.na(j) && !all(cs$conductanceMatrix[, j] == 0), logical(1))
cat("anchors:", nrow(anc), "; enterable", sum(ok), "; dropped", paste(anc$id[!ok], collapse = " "), "\n")
locs <- locs[ok, ]
if (nrow(locs) < 2) {
  write_paths(NULL, out_f)
  cat("paths: 0 (fewer than 2 enterable anchors)\n")
  quit(status = 0)
}
lcps <- create_FETE_lcps(x = cs, locations = locs, cost_distance = FALSE, ncores = 1)
# map the package's origin / destination row numbers back to anchor ids
ids <- anc$id[ok]
lcps$origin_ID <- ids[lcps$origin_ID]
lcps$destination_ID <- ids[lcps$destination_ID]
n <- write_paths(lcps, out_f)
cat("paths:", n, "of", length(ids) * (length(ids) - 1), "ordered pairs; seconds",
    round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "\n")
