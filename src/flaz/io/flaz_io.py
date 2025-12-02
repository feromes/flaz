from urllib.parse import urlparse
import pyarrow.parquet as pq
from pathlib import Path
import os

PROJECT_TEMP_DIR = Path.cwd() / "flaz_tmp"

class FlazIO:

    def write_favela(self, favela, uri: str) -> str:
        parsed = urlparse(uri)

        # 1) Esquema temporário controlado pelo projeto
        if parsed.scheme == "temp":
            # ex.: "temp://sao_remo_2017.parquet"
            # parsed.netloc = "sao_remo_2017.parquet"
            # parsed.path   = "" (ou "/algo", se você usar subpastas)
            rel_path = (parsed.netloc + parsed.path).lstrip("/")
            PROJECT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            uri_path = PROJECT_TEMP_DIR / rel_path

        # 2) Esquema file:// → path local
        elif parsed.scheme == "file":
            uri_path = Path(parsed.path)

        # 3) Caso contrário, tratar como path diretamente
        else:
            uri_path = Path(uri)

        # Garantir diretório, se tiver
        if uri_path.parent:
            uri_path.parent.mkdir(parents=True, exist_ok=True)

        # Salvar Parquet puro
        pq.write_table(favela.table, uri_path.as_posix())

        # Retornar o caminho final para quem chamar
        return uri_path.as_posix()

    

    
