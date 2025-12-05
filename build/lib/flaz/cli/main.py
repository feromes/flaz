import typer
from flaz import Favela

app = typer.Typer(help="CLI do projeto FLAZ")

@app.command()
def calc_hag(
    favela: str = typer.Option(..., "--favela", "-f", help="Nome da favela"),
    ano: int = typer.Option(..., "--ano", "-a", help="Ano do processamento"),
    force: bool = typer.Option(False, "--force", help="Recalcula mesmo com cache"),
):
    """
    Calcula a camada HAG para uma favela específica.
    """
    typer.echo(f"Calculando HAG para {favela} ({ano})...")

    f = Favela(favela).periodo(ano)
    f.calc_hag()

    out = f"temp://{favela}_{ano}_hag.arrow"
    f.persist(out)

    typer.echo(f"Concluído! Arquivo salvo em {out}")

def run():
    app()

