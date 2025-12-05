from urllib.parse import urlparse
from pathlib import Path
import os
import pyarrow as pa
import pyarrow.ipc as ipc

PROJECT_TEMP_DIR = Path.cwd() / "flaz_tmp"

# --- opcional: suporte ao Backblaze B2 ---
try:
    from b2sdk.v2 import InMemoryAccountInfo, B2Api
    B2_AVAILABLE = True
except ImportError:
    B2_AVAILABLE = False


class FlazIO:

    def write_favela(self, favela, uri: str) -> str:
        """
        Persiste uma favela em formato Arrow IPC (único formato oficial do FLAZ).
        Suporta:
        - temp://arquivo.arrow
        - file:///caminho/arquivo.arrow
        - b2://bucket/arquivo.arrow
        - caminho_local.arrow
        """

        table = favela.table
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()

        # 1) TEMP://  → salvar em flaz_tmp/
        if scheme == "temp":
            rel = (parsed.netloc + parsed.path).lstrip("/")
            dest = PROJECT_TEMP_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._write_arrow(table, dest)
            return dest.as_posix()

        # 2) FILE:// → salvar localmente
        elif scheme == "file":
            dest = Path(parsed.path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._write_arrow(table, dest)
            return dest.as_posix()

        # 3) B2:// → enviar Arrow para Backblaze B2
        elif scheme == "b2":
            if not B2_AVAILABLE:
                raise RuntimeError(
                    "b2sdk não instalado. Execute: pip install b2sdk"
                )

            bucket_name = parsed.netloc
            key = parsed.path.lstrip("/")

            binary = self._arrow_bytes(table)
            return self._upload_b2(bucket_name, key, binary)

        # 4) Caso sem esquema → tratar como path local
        else:
            dest = Path(uri)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._write_arrow(table, dest)
            return dest.as_posix()



    # --------------------------------------------------------------
    # ARROW — Formato único do FLAZ
    # --------------------------------------------------------------

    def _write_arrow(self, table, dest: Path):
        """Grava em Arrow IPC (arquivo .arrow)."""
        with pa.OSFile(dest.as_posix(), "wb") as f:
            with ipc.RecordBatchFileWriter(f, table.schema) as writer:
                writer.write_table(table)

    def _arrow_bytes(self, table) -> bytes:
        """Retorna o conteúdo Arrow IPC em bytes (para B2, R2 etc)."""
        sink = pa.BufferOutputStream()
        with ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()


    # --------------------------------------------------------------
    # BACKBLAZE B2
    # --------------------------------------------------------------

    def _upload_b2(self, bucket_name: str, key: str, data: bytes) -> str:
        info = InMemoryAccountInfo()
        b2 = B2Api(info)

        app_key_id = os.environ["B2_KEY_ID"]
        app_key = os.environ["B2_APP_KEY"]

        b2.authorize_account("production", app_key_id, app_key)
        bucket = b2.get_bucket_by_name(bucket_name)

        uploaded = bucket.upload_bytes(
            data,
            key,
            content_type="application/octet-stream"
        )

        # URL pública final
        return f"{bucket.get_download_url(base_name=False)}/{key}"
