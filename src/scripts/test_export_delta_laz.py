from pathlib import Path

from flaz import Favela


def test_export_delta_laz_sao_remo():
    """
    Script de teste completo:
    - Garante base e flaz de 2017
    - Garante base e flaz de 2020
    - Calcula delta REAL (2020 - 2017)
    - Força reescrita do delta_flaz.copc.laz
    """

    # ----------------------------------------
    # CONFIG
    # ----------------------------------------
    API_PATH = Path("./flaz_api").expanduser().resolve()
    FAVELA = "São Remo"

    ANO_REF = 2017
    ANO_ATUAL = 2020

    print("🚀 Teste calc + export delta_flaz.copc.laz — São Remo")

    # ============================================================
    # 1. ANO DE REFERÊNCIA (2017)
    # ============================================================
    print(f"\n🕰️ Preparando ano de referência: {ANO_REF}")

    f_ref = (
        Favela(FAVELA)
        .set_api_path(API_PATH)
        .periodo(ANO_REF)
    )

    print("→ garantindo base LiDAR (COPC)")
    f_ref._build_favela_lidar_base(
        out_dir=f_ref.periodo_dir(ANO_REF),
        force=False
    )

    print("→ calculando flaz (arrow)")
    f_ref.calc_flaz()
    f_ref.persist(API_PATH)

    print(f"→ 2017 pronto ({f_ref.table.num_rows} pontos)")

    # ============================================================
    # 2. ANO ATUAL (2020)
    # ============================================================
    print(f"\n🕰️ Preparando ano atual: {ANO_ATUAL}")

    f = (
        Favela(FAVELA)
        .set_api_path(API_PATH)
        .periodo(ANO_ATUAL)
    )

    print("→ garantindo base LiDAR (COPC)")
    f._build_favela_lidar_base(
        out_dir=f.periodo_dir(ANO_ATUAL),
        force=False
    )

    print("→ calculando flaz (arrow)")
    f.calc_flaz()
    f.persist(API_PATH)

    print(f"→ 2020 pronto ({f.table.num_rows} pontos)")

    # ============================================================
    # 3. CALCULA DELTA REAL (2020 - 2017)
    # ============================================================
    print("\n📐 Calculando delta morfológico (2020 - 2017)")

    f.calc_delta(
        ano_ref=ANO_REF,
        raio_m=1.0,
        min_vizinhos=5,
        stat="median",
    )

    # ============================================================
    # 4. EXPORTA (FORÇANDO REESCRITA)
    # ============================================================
    print("\n💾 Exportando delta_flaz.copc.laz (force overwrite)")

    out_path = f.export_delta_laz(force=True)

    print("\n✅ Artefato delta gerado com sucesso:")
    print(out_path)


if __name__ == "__main__":
    test_export_delta_laz_sao_remo()
