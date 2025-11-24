# FLAZ — Biblioteca de ETL Espacial 3D+t para o FavelaVIZ

Formato leve, modular e declarativo para dados urbanos de favelas

## Visão Geral

FLAZ é uma biblioteca Python para processamento, organização e indexação de dados espaciais 3D+t, criada para o ecossistema FavelaVIZ e preparada para integrar o futuro OGDC (Open GeoData Cube).

Seu propósito é transformar nuvens de pontos (LAZ/COPC) e derivados em um formato modular, científico e visualmente eficiente, usando:

- PyArrow/GeoArrow
- índice Morton 3D/4D + era temporal
- arquitetura em camadas
- artefatos .bin otimizados para WebGPU/WebGL
- computação distribuída com Ray
- transformações puras encadeáveis

## Objetivos Centrais

1. Padronizar um formato universal para dados 3D+t de favelas
1. Produzir camadas independentes (HAG, SVF, vielas, grafos, rasters…)
1. Garantir alinhamento espacial-temporal com flaz_index (96–128 bits)
1. Gerar artefatos leves para renderização web (.bin)
1. Integrar diretamente com o frontend @fviz (Next.js + React + WebGPU)
1. Permitir ETL escalável e distribuído com Ray
1. Ser simples para o usuário e rigoroso para a ciência

## Arquitetura Conceitual

O FLAZ organiza o mundo urbano com três entidades:

### Favelas
- Representa um conjunto de favelas
- Executa ETL distribuído (calc_all(workers=N))
- Resolve clusters (“Grajaú”, “Butantã”, …)

### Favela

- Unidade de processamento
- Carrega períodos/épocas
- Converte LAZ/COPC → Arrow
- Executa transformações (calc_hag(), calc_vielas(), …)
- Salva camadas modulares

### FViz

- Objeto universal de retorno para cada cálculo

> Guarda:
  - Geometria (geom)
  - Métricas específicas
  - fviz.json (para renderização)
  - .bin (GPU-friendly)
  - Metadata

>É a ponte final entre ciências 3D+t e visualização web

## Instalação

```bash
pip install flaz
```

### Exemplo de Uso

#### Carregando uma favela e calculando HAG

```Python
from flaz import Favela

fl = Favela("São Remo").periodo(2017)

res = fl.calc_hag()
res.save()

print(res.geom)
```

#### Extraindo vielas

```Python
res = fl.calc_vielas(w_max=6.0)

print("Comprimento total:", res.length)
res.save()       # salva parquet + bin + fviz.json + meta
```

#### Processando um cluster inteiro (distribuído)

```Python
from flaz import Favelas

Favelas("Butantã").calc_all(workers=64)
```

### Estrutura do Formato FLÁZ

Um diretório FLAZ típico segue o padrão:

```Python
Favela_Ano/
    flaz.points.parquet          # base científica (x,y,z,class,flaz_index)
    flaz.metadata.json

    flaz.hag.parquet             # feature por ponto
    flaz.normals.parquet

    flaz.vielas.parquet          # feature geométrica
    flaz.vielas.bin              # runtime FViz
    flaz.vielas.fviz.json
    flaz.vielas.meta.json

    flaz.svf_points.parquet
    flaz.svf_raster.tif
    flaz.svf.meta.json
```

Cada arquivo é indexado pelo `flaz_index`, garantindo alinhamento perfeito entre camadas.

## Princípios da Biblioteca

1. OO Semântico

    Classes representam ideias urbanas reais, não apenas estruturas técnicas.

2. Transformações Puras

    `calc_*` nunca altera estado; sempre retorna um FViz completo.

3. Camadas Modulares

    Cada feature tem seu próprio arquivo.
    FLAZ é um data cube espacial-temporal, não um arquivo monolítico.

4. Runtime Leve

    `.bin` é otimizado para WebGL/WebGPU:
    >compacto, rápido e ideal para visualização.

## Desenvolvimento e Testes

Instalação em modo editável
```
pip install -e .
```

## Testes (em construção)

`pytest`

## Roadmap

- 🔲 Conversão LAZ/COPC → Arrow
- 🔲 Implementação das features 1:
    - 🔲 MDT
    - 🔲 MDS
    - 🔲 Edificações (footprint)
    - 🔲 HAG
    - 🔲 Vielas
- 🔲 Implementação das features 2:
    - 🔲 SVF
    - 🔲 Espaços livres e ventilação
    - 🔲 Cabeamento aéreo
    - 🔲 Vegetação
    - 🔲 Grafos (kNN)
    - 🔲 Embedings
    - 🔲 Insolação
    - 🔲 Campos (risk field, density field)
- 🔲 Geração automática de `.bin`+ ``fviz.json``
- 🔲 CLI: ```flaz build <favela> <ano>```
- 🔲 Integração completa com Ray
- 🔲 Suporte a .bin.zst e streaming
- 🔲 Documentação no Docusaurus
- 🔲 Publicação no PyPI

## 🤝 Contribuições

Contribuições serão bem-vindas quando o projeto alcançar a versão 1.0.
Sinta-se livre para abrir issues, propor features e sugerir melhorias, copiar, modificar executar.

## Licença

A definir — provavelmente MIT.