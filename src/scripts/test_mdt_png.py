from pathlib import Path
from flaz import Favela


def main():
    nome = "São Remo"
    ano = 2017

    out_root = Path("flaz_tmp")

    print("Favela:", nome)
    print("Ano:", ano)
    print("Root:", out_root.resolve())

    f = Favela(nome).periodo(ano)

    # ------------------------------------------------------------
    # 1. Build da base LiDAR (COPC + MDS + MDT.tif)
    # ------------------------------------------------------------
    out_dir = (
        out_root
        / "favela"
        / f.nome_normalizado()
        / "periodos"
        / str(ano)
    )

    print("\n▶ Construindo base LiDAR…")
    f._build_favela_lidar_base(out_dir)

    # ------------------------------------------------------------
    # 2. Persistência (gera PNG + JSON)
    # ------------------------------------------------------------
    print("\n▶ Persistindo artefatos (PNG + metadata)…")
    f.persist(out_root)

    # ------------------------------------------------------------
    # 3. Verificações
    # ------------------------------------------------------------
    mdt_tif = out_dir / "mdt.tif"
    mdt_png = out_dir / "mdt.png"
    mdt_json = out_dir / "mdt.json"

    print("\n▶ Verificação de arquivos:")

    for p in [mdt_tif, mdt_png, mdt_json]:
        status = "OK" if p.exists() else "❌ NÃO EXISTE"
        print(f" - {p.name}: {status}")

    print("\n✅ Teste finalizado.")


if __name__ == "__main__":
    main()
