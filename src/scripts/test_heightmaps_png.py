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

    out_dir = (
        out_root
        / "favela"
        / f.nome_normalizado()
        / "periodos"
        / str(ano)
    )

    # ------------------------------------------------------------
    # 1. Build da base LiDAR
    # ------------------------------------------------------------
    print("\n▶ Construindo base LiDAR…")
    f._build_favela_lidar_base(out_dir)

    # ------------------------------------------------------------
    # 2. Persistência (gera PNGs + JSONs)
    # ------------------------------------------------------------
    print("\n▶ Persistindo artefatos (MDT + MDS)…")
    f.persist(out_root)

    # ------------------------------------------------------------
    # 3. Verificação
    # ------------------------------------------------------------
    files = [
        out_dir / "mdt.tif",
        out_dir / "mdt.png",
        out_dir / "mdt.json",
        out_dir / "mds.tif",
        out_dir / "mds.png",
        out_dir / "mds.json",
    ]

    print("\n▶ Verificação de arquivos:")
    for p in files:
        status = "OK" if p.exists() else "❌ NÃO EXISTE"
        print(f" - {p.name}: {status}")

    print("\n✅ Teste finalizado.")


if __name__ == "__main__":
    main()
