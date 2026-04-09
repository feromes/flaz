# src/flaz/articles/growth/process_hag_one.py

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from flaz import Favela

import csv
import numpy as np
import rasterio

def compute_hag_stats_and_histogram(
    hag_path: Path,
    out_csv: Path,
    favela_nome: str,
    year: int,
):
    with rasterio.open(hag_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    arr = arr.astype("float64")

    valid_mask = np.isfinite(arr)
    if nodata is not None:
        valid_mask &= arr != nodata

    values = arr[valid_mask]

    if values.size == 0:
        raise ValueError(f"Raster sem pixels válidos: {hag_path}")

    n_valid = int(values.size)
    sum_hag = float(values.sum())
    mean_hag = float(values.mean())
    median_hag = float(np.median(values))
    std_hag = float(values.std())

    min_hag = float(values.min())
    max_hag = float(values.max())

    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))

    # Histograma apenas para HAG >= 2m
    values_2m = values[values >= 2.0]

    n_pixels_ge_2m = int(values_2m.size)
    prop_pixels_ge_2m = float(n_pixels_ge_2m / n_valid)

    sum_hag_ge_2m = float(values_2m.sum()) if n_pixels_ge_2m > 0 else 0.0

    # 99 bins entre 2 e 20
    edges = np.linspace(2.0, 20.0, 100)  # 100 edges -> 99 bins
    hist_99, _ = np.histogram(values_2m[values_2m < 20.0], bins=edges)

    # bin 100 = tudo >= 20m
    bin_100 = int((values_2m >= 20.0).sum())

    hist_100 = list(hist_99.astype(int)) + [bin_100]

    row = {
        "favela": favela_nome,
        "year": year,

        # base
        "n_valid_pixels": n_valid,
        "sum_hag": sum_hag,
        "min_hag": min_hag,
        "mean_hag": mean_hag,
        "median_hag": median_hag,
        "std_hag": std_hag,
        "max_hag": max_hag,

        # quantis
        "p25_hag": p25,
        "p75_hag": p75,
        "p90_hag": p90,
        "p95_hag": p95,

        # construção (>=2m)
        "n_pixels_ge_2m": n_pixels_ge_2m,
        "prop_pixels_ge_2m": prop_pixels_ge_2m,
        "sum_hag_ge_2m": sum_hag_ge_2m,
    }

    for i, count in enumerate(hist_100, start=1):
        row[f"bin_{i:03d}"] = count

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"[ok] Estatísticas salvas em: {out_csv}")

def compute_mdt_stats(
    mdt_path: Path,
    out_csv: Path,
    favela_nome: str,
    year: int,
):
    with rasterio.open(mdt_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        res_x, res_y = src.res

    valid_mask = np.isfinite(arr)
    if nodata is not None:
        valid_mask &= arr != nodata

    values = arr[valid_mask]

    if values.size == 0:
        raise ValueError(f"Raster sem pixels válidos: {mdt_path}")

    # estatísticas básicas
    n_valid = int(values.size)
    sum_mdt = float(values.sum())
    min_mdt = float(values.min())
    mean_mdt = float(values.mean())
    median_mdt = float(np.median(values))
    std_mdt = float(values.std())
    max_mdt = float(values.max())

    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))

    relief_amplitude = float(max_mdt - min_mdt)

    # declividade
    arr_valid = arr.copy()

    if nodata is not None:
        arr_valid[arr_valid == nodata] = np.nan

    # gradientes em m/m
    gy, gx = np.gradient(arr_valid, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad)

    slope_values = slope_deg[np.isfinite(slope_deg)]

    if slope_values.size == 0:
        slope_mean = np.nan
        slope_median = np.nan
        slope_std = np.nan
        slope_p90 = np.nan
        slope_p95 = np.nan
    else:
        slope_mean = float(slope_values.mean())
        slope_median = float(np.median(slope_values))
        slope_std = float(slope_values.std())
        slope_p90 = float(np.percentile(slope_values, 90))
        slope_p95 = float(np.percentile(slope_values, 95))

    row = {
        "favela": favela_nome,
        "year": year,
        "n_valid_pixels": n_valid,
        "sum_mdt": sum_mdt,
        "min_mdt": min_mdt,
        "mean_mdt": mean_mdt,
        "median_mdt": median_mdt,
        "std_mdt": std_mdt,
        "max_mdt": max_mdt,
        "p25_mdt": p25,
        "p75_mdt": p75,
        "p90_mdt": p90,
        "p95_mdt": p95,
        "relief_amplitude": relief_amplitude,
        "slope_mean_deg": slope_mean,
        "slope_median_deg": slope_median,
        "slope_std_deg": slope_std,
        "slope_p90_deg": slope_p90,
        "slope_p95_deg": slope_p95,
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"[ok] Estatísticas do MDT salvas em: {out_csv}")

def build_hag_raster(
    favela_nome: str,
    year: int,
    api_path: str | Path,
    resolution: float = 0.5,
    force: bool = False,
) -> Path:
    """
    Gera apenas o HAG raster de uma favela em um ano.

    Fluxo:
    1. Instancia Favela
    2. Define api_path
    3. Define período
    4. Garante a base LiDAR (inclusive favela.copc.laz)
    5. Rasteriza HeightAboveGround em hag_050.tif
    """

    api_path = Path(api_path).expanduser().resolve()

    f = Favela(favela_nome)
    f.set_api_path(api_path)
    f.periodo(year)

    out_dir = f.periodo_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    copc_path = out_dir / "favela.copc.laz"
    hag_path = out_dir / "hag_050.tif"
    pipeline_path = out_dir / "favela_hag_pipeline.json"

    mdt_path = out_dir / "mdt.tif"
    mdt_stats_csv = out_dir / "mdt_stats.csv"

    if hag_path.exists() and not force:
        print(f"[skip] HAG já existe: {hag_path}")
        return hag_path

    # 1) Garante a base LiDAR da favela/ano
    if not copc_path.exists() or force:
        print(f"[info] Construindo base LiDAR para {favela_nome} ({year})...")
        f._build_favela_lidar_base(out_dir=out_dir, force=force)

    if not mdt_path.exists():
        raise FileNotFoundError(f"MDT não encontrado após build da base: {mdt_path}")

    # 2) Define grid alinhado
    grid = f._compute_aligned_grid(resolution)

    # 3) Pipeline PDAL: lê o COPC já com HeightAboveGround e escreve raster HAG
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": str(copc_path),
            },
            {
                "type": "filters.range",
                "limits": "Classification[6:6]",
            },
            {
                "type": "writers.gdal",
                "filename": str(hag_path),
                "dimension": "HeightAboveGround",
                "resolution": grid["resolution"],
                "origin_x": grid["origin_x"],
                "origin_y": grid["origin_y"],
                "width": grid["width"],
                "height": grid["height"],
                "output_type": "max",
                "nodata": -9999,
                "override_srs": "EPSG:31983",
            },
        ]
    }

    pipeline_path.write_text(
        json.dumps(pipeline, indent=2),
        encoding="utf-8",
    )

    print(f"[info] Gerando HAG raster: {hag_path}")
    subprocess.run(
        ["pdal", "pipeline", str(pipeline_path)],
        check=True,
    )

    stats_csv = out_dir / "hag_stats.csv"

    compute_hag_stats_and_histogram(
        hag_path=hag_path,
        out_csv=stats_csv,
        favela_nome=favela_nome,
        year=year,
    )

    compute_mdt_stats(
        mdt_path=mdt_path,
        out_csv=mdt_stats_csv,
        favela_nome=favela_nome,
        year=year,
    )

    print(f"[ok] HAG gerado em: {hag_path}")
    return hag_path


def main():
    parser = argparse.ArgumentParser(
        description="Gera apenas o HAG raster de uma favela/ano."
    )
    parser.add_argument("--favela", required=True, help="Nome da favela")
    parser.add_argument("--year", required=True, type=int, help="Ano: 2017, 2020 ou 2024")
    parser.add_argument("--api", required=True, help="Caminho da pasta flaz_api")
    parser.add_argument("--resolution", type=float, default=0.5, help="Resolução do raster")
    parser.add_argument("--force", action="store_true", help="Reprocessa mesmo se já existir")

    args = parser.parse_args()

    build_hag_raster(
        favela_nome=args.favela,
        year=args.year,
        api_path=args.api,
        resolution=args.resolution,
        force=args.force,
    )


if __name__ == "__main__":
    main()