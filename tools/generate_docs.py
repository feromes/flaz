import inspect
import importlib
from pathlib import Path

MODULES = [
    "flaz.models.favela",
    "flaz.models.favelas",
    "flaz.models.fviz",
]

ROOT = Path(__file__).resolve().parent.parent  # volta para raiz do projeto
OUTPUT_DIR = ROOT / "docs" / "api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def doc_for_module(module_name):
    module = importlib.import_module(module_name)
    out = [f"# {module_name}", ""]

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            out.append(f"## Class {name}")
            out.append("")
            out.append(inspect.getdoc(obj) or "*Sem docstring.*")
            out.append("")

            for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                out.append(f"### Method {name}.{method_name}")
                out.append("")
                out.append("```python")
                out.append(f"{name}.{method_name}()")
                out.append("```")
                out.append("")
                out.append(inspect.getdoc(method) or "*Sem docstring.*")
                out.append("")

    return "\n".join(out)

if __name__ == "__main__":
    for m in MODULES:
        md = doc_for_module(m)
        fname = m.replace(".", "_") + ".md"
        (OUTPUT_DIR / fname).write_text(md)
        print(f"[OK] gerado: docs/api/{fname}")
