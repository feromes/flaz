import typer
from flaz import Favela, Favelas

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def calc_hag(
    favela: str = typer.Option(..., "--favela", "-f", help="Nome da favela."),
    ano: int = typer.Option(..., "--ano", "-a", help="Ano do processamento."),
    force: bool = typer.Option(False, "--force", help="Ignora cache."),
):
    """
    Calcula a camada HAG para uma única favela.
    """
    typer.echo(f"Calculando HAG para {favela} ({ano})...")

    f = Favela(favela).periodo(ano)
    f.calc_flaz()
    out = f"temp://{favela}_{ano}_hag.arrow"
    f.persist(out)

    typer.echo(f"✔ Concluído! Arquivo salvo em {out}")

@app.command()
def calc_more(
    ano: int = typer.Option(..., "--ano", "-a", help="Ano do processamento."),
    force: bool = typer.Option(False, "--force", help="Ignora cache."),
):
    """
    Processa todas as favelas listadas em Favelas.FAVELAS_MORE.
    """
    nomes = Favelas.FAVELAS_MORE

    typer.echo(f"Processando {len(nomes)} favelas (FAVELAS_MORE)...\n")

    for nome in nomes:
        typer.echo(f"→ {nome} ({ano})")

        f = Favela(nome).periodo(ano)
        f.calc_flaz()   # ou calc_hag se preferir, você escolhe
        out = f"temp://{nome}_{ano}_hag.arrow"
        f.persist(out)

    typer.echo("\n✔ Concluído processamento de todas as favelas!")


if __name__ == "__main__":
    app()
