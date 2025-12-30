import typer
from flaz import Favela, Favelas
from pathlib import Path
import warnings
import json

warnings.filterwarnings(
    "ignore",
    message="Measured \\(M\\) geometry types are not supported.*",
    category=UserWarning,
    module="pyogrio"
)


app = typer.Typer(pretty_exceptions_enable=False)

def resolve_api_path(api: str) -> Path:
    """
    Resolve o caminho da API como Path gravável.
    Aceita caminho relativo ou absoluto.
    """
    return Path(api).expanduser().resolve()

@app.command()
def calc_hag(
    favela: str = typer.Option(..., "--favela", "-f", help="Nome da favela."),
    ano: int = typer.Option(..., "--ano", "-a", help="Ano do processamento."),
    api: str = typer.Option(
        "./flaz_api",
        "--api",
        help="Diretório raiz onde a API FLAZ será gravada."
    ),
    force: bool = typer.Option(False, "--force", help="Ignora cache."),
):
    """
    Calcula a camada HAG para uma única favela.
    """
    api_path = resolve_api_path(api)

    typer.echo(f"→ API path: {api_path}")

    f = Favela(favela)
    f.periodo(ano).calc_flaz()
    f.persist(api_path)

    card = f.to_card()

    card_path = api_path / "favelas.json"
    card_path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    typer.echo("✔ Concluído!")

@app.command()
def calc_more(
    ano: int = typer.Option(..., "--ano", "-a", help="Ano do processamento."),
    api: str = typer.Option(
        "./flaz_api",
        "--api",
        help="Diretório raiz onde a API FLAZ será gravada."
    ),
    force: bool = typer.Option(False, "--force", help="Ignora cache."),
):
    """
    Processa todas as favelas listadas em Favelas.FAVELAS_MORE.
    """
    favelas = Favelas()

    typer.echo(f"Processando {len(favelas)} favelas")

    cards = []

    for f in favelas:
        api_path = resolve_api_path(api)

        f.periodo(ano)
        typer.echo(f"→ API path: {api_path}")
        typer.echo(f"→ {f} ({ano})")

        f.calc_flaz()
        f.persist(api_path)

        cards.append(f.to_card())

    # cards = Favelas().to_cards()

    cards_path = api_path / "favelas.json"
    cards_path.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # # Gera o json de Favelas.More dentro da API
    # favelas = Favelas()

    # root = Path(api)
    # json_out = root / "favelas.json"
    # json_out.parent.mkdir(parents=True, exist_ok=True)

    # json_out.write_text(
    #     favelas.to_json(),
    #     encoding="utf-8"
    # )

    # typer.echo(f"\n✔ Arquivo JSON salvo em {json_out}")
    # typer.echo("✔ Concluído processamento de todas as favelas!")

