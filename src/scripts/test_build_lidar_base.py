from pathlib import Path
from flaz import Favela


def main():
    # ------------------------------------------------------------
    # Configuração básica
    # ------------------------------------------------------------
    out_root = Path("flaz_tmp")
    out_root.mkdir(exist_ok=True)

    favela_nome = "São Remo"
    ano = 2017

    # ------------------------------------------------------------
    # Instancia a Favela
    # ------------------------------------------------------------
    f = (
        Favela(favela_nome)
        .periodo(ano)
    )

    print("Favela:", f.nome)
    print("Ano:", ano)

    # ------------------------------------------------------------
    # Diretório de saída
    # ------------------------------------------------------------
    out_dir = out_root / "favela" / f.nome_normalizado() / "periodos" / str(ano)

    print("Output dir:")
    print(out_dir.resolve())

    # ------------------------------------------------------------
    # Build da base LiDAR
    # ------------------------------------------------------------
    print("\n▶ Construindo base LiDAR da favela...\n")

    result = f._build_favela_lidar_base(
        out_dir=out_dir,
        force=True,   # 👈 sempre recalcula durante testes
    )

    # ------------------------------------------------------------
    # Resultado
    # ------------------------------------------------------------
    print("\n✔ Artefatos gerados:\n")
    for k, v in result.items():
        print(f"- {k}: {v}")

    print("\n✅ Teste concluído com sucesso.")


if __name__ == "__main__":
    main()
