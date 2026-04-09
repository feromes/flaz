from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd


import re
from pathlib import Path
from typing import List, Tuple


def find_favela_year_dirs(favela_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Retorna tuplas (favela, ano, pasta_ano) procurando por:

    <favela_root>/**/periodos/{ano}/

    Ex.: 
    - favela/abacateiro/periodos/2020/
    - favela/alcindo_ferreira_i_/_jardim_cruzeiro/periodos/2024/
    """
    items: List[Tuple[str, str, Path]] = []

    if not favela_root.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {favela_root}")

    for periodos_dir in sorted(p for p in favela_root.rglob("periodos") if p.is_dir()):
        rel_parent = periodos_dir.relative_to(favela_root).parent
        favela = str(rel_parent).replace("/", "__")

        for year_dir in sorted(p for p in periodos_dir.iterdir() if p.is_dir()):
            ano = year_dir.name

            if re.fullmatch(r"\d{4}", ano):
                items.append((favela, ano, year_dir))

    return items


def read_stats_csv(csv_path: Path) -> Tuple[pd.DataFrame | None, str, int]:
    """
    Lê um CSV e retorna:
    - dataframe (ou None)
    - status: ok | missing | empty | error
    - n_rows
    """
    if not csv_path.exists():
        return None, "missing", 0

    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            return df, "empty", 0

        return df, "ok", len(df)

    except Exception:
        return None, "error", 0


def collect_one_type(
    favela_year_dirs: List[Tuple[str, str, Path]],
    filename: str,
    tipo: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coleta e concatena um tipo de CSV (hag_stats.csv ou mdt_stats.csv).

    Retorna:
    - tabelão concatenado
    - auditoria
    """
    all_rows: List[pd.DataFrame] = []
    audit_rows: List[Dict] = []

    for favela, ano, year_dir in favela_year_dirs:
        csv_path = year_dir / filename
        df, status, n_rows = read_stats_csv(csv_path)

        audit_rows.append(
            {
                "favela": favela,
                "ano": ano,
                "tipo": tipo,
                "arquivo": filename,
                "caminho": str(csv_path),
                "status": status,
                "n_rows": n_rows,
            }
        )

        if status == "ok" and df is not None:
            df = df.copy()

            # remove colunas preexistentes para recolocar na frente com valor padronizado
            if "favela" in df.columns:
                df = df.drop(columns=["favela"])
            if "ano" in df.columns:
                df = df.drop(columns=["ano"])

            df.insert(0, "favela", favela)
            df.insert(1, "ano", ano)
            all_rows.append(df)

    if all_rows:
        big_table = pd.concat(all_rows, ignore_index=True)
    else:
        big_table = pd.DataFrame()

    audit_table = pd.DataFrame(audit_rows)
    return big_table, audit_table


def build_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera um resumo agregado da auditoria por tipo de arquivo.
    """
    if audit_df.empty:
        return pd.DataFrame()

    summary = (
        audit_df.groupby(["tipo", "status"], dropna=False)
        .size()
        .reset_index(name="quantidade")
        .sort_values(["tipo", "status"])
    )
    return summary


def build_missing_matrix(hag_audit: pd.DataFrame, mdt_audit: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma matriz única por favela/ano indicando situação de HAG e MDT.
    """
    hag_cols = ["favela", "ano", "status", "n_rows"]
    mdt_cols = ["favela", "ano", "status", "n_rows"]

    hag = hag_audit[hag_cols].rename(
        columns={"status": "hag_status", "n_rows": "hag_n_rows"}
    )
    mdt = mdt_audit[mdt_cols].rename(
        columns={"status": "mdt_status", "n_rows": "mdt_n_rows"}
    )

    merged = hag.merge(mdt, on=["favela", "ano"], how="outer")

    def classify(row: pd.Series) -> str:
        hag_ok = row.get("hag_status") == "ok"
        mdt_ok = row.get("mdt_status") == "ok"

        if hag_ok and mdt_ok:
            return "ok"
        if (not hag_ok) and (not mdt_ok):
            return "missing_both_or_invalid"
        if not hag_ok:
            return "problem_hag"
        return "problem_mdt"

    merged["auditoria_geral"] = merged.apply(classify, axis=1)
    return merged.sort_values(["favela", "ano"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatena hag_stats.csv e mdt_stats.csv e gera auditoria."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Pasta raiz que contém flaz_api/favela/",
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()

    # saída fixa dentro do projeto
    out_dir = (root / "flaz" / "dinamica_favelas").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)              

    candidate_paths = [
        root / "flaz_api" / "favela",
        root / "flaz" / "flaz_api" / "favela",
        root / "data" / "flaz_api" / "favela",
    ]

    favela_root = None
    for p in candidate_paths:
        if p.exists():
            favela_root = p
            break

    if favela_root is None:
        raise FileNotFoundError(
            "Nenhuma pasta de favelas foi encontrada. Caminhos testados:\n"
            + "\n".join(f" - {p}" for p in candidate_paths)
        )

    print(f"[info] Pasta de entrada encontrada: {favela_root}")
    favela_year_dirs = find_favela_year_dirs(favela_root)

    # HAG
    hag_big, hag_audit = collect_one_type(
        favela_year_dirs=favela_year_dirs,
        filename="hag_stats.csv",
        tipo="hag",
    )

    # MDT
    mdt_big, mdt_audit = collect_one_type(
        favela_year_dirs=favela_year_dirs,
        filename="mdt_stats.csv",
        tipo="mdt",
    )

    # Auditoria consolidada
    audit_all = pd.concat([hag_audit, mdt_audit], ignore_index=True)
    summary = build_summary(audit_all)
    missing_matrix = build_missing_matrix(hag_audit, mdt_audit)

    # Saídas
    hag_big.to_csv(out_dir / "tabelao_hag_stats.csv", index=False)
    mdt_big.to_csv(out_dir / "tabelao_mdt_stats.csv", index=False)

    hag_audit.to_csv(out_dir / "auditoria_hag_stats.csv", index=False)
    mdt_audit.to_csv(out_dir / "auditoria_mdt_stats.csv", index=False)
    audit_all.to_csv(out_dir / "auditoria_todos.csv", index=False)
    summary.to_csv(out_dir / "auditoria_resumo.csv", index=False)
    missing_matrix.to_csv(out_dir / "auditoria_favela_ano.csv", index=False)

    # Impressão de resumo
    total_pairs = len(favela_year_dirs)
    print(f"\n[info] Total de combinações favela/ano encontradas: {total_pairs}")

    print("\n[info] Resumo da auditoria:")
    if summary.empty:
        print("Nenhum dado encontrado.")
    else:
        print(summary.to_string(index=False))

    print("\n[info] Arquivos gerados:")
    print(f" - {out_dir / 'tabelao_hag_stats.csv'}")
    print(f" - {out_dir / 'tabelao_mdt_stats.csv'}")
    print(f" - {out_dir / 'auditoria_hag_stats.csv'}")
    print(f" - {out_dir / 'auditoria_mdt_stats.csv'}")
    print(f" - {out_dir / 'auditoria_todos.csv'}")
    print(f" - {out_dir / 'auditoria_resumo.csv'}")
    print(f" - {out_dir / 'auditoria_favela_ano.csv'}")


if __name__ == "__main__":
    main()