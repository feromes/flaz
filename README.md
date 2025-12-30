# FLAZ — Biblioteca de ETL Espacial 3D+t para o FavelaVIZ

Formato leve, modular e declarativo para dados urbanos de favelas. Ferramenta de trabalho para o Doutorado do autor: Fernando Gomes para transformar processos em linhas de comando padronizadas e formatos compatíveis com diversos usos

## Visão Geral

FLAZ é uma biblioteca Python para processamento, organização e indexação de dados espaciais 3D+t, criada para o ecossistema FZ (FavelaVIZ) e preparada para integrar o futuro OGDC (Open GeoData Cube).

Seu propósito é transformar nuvens de pontos (LAZ/COPC) e derivados em um formato modular, científico e visualmente eficiente, usando:

- PyArrow/GeoArrow
- índice Morton 3D/4D + era temporal
- arquitetura em camadas
- artefatos .arrow ou .flaz otimizados para WebGPU/WebGL
- computação distribuída com Ray
- transformações puras encadeáveis atravéz de plugins

## Objetivos Centrais

1. Padronizar um formato universal para dados 3D+t de favelas
1. Produzir camadas independentes (HAG, SVF, vielas, grafos, rasters…)
1. Garantir alinhamento espacial-temporal com flaz_index (96–128 bits)
1. Gerar artefatos leves para renderização web (.bin, .flaz, .arrow, .json)
1. Integrar diretamente com o frontend @fviz (Next.js + React + Three.js)
1. Permitir ETL escalável e distribuído com Ray
1. Ser simples para o usuário e rigoroso para a ciência

## Arquitetura Conceitual

O FLAZ organiza o mundo urbano com três entidades:

### Favelas
- Representa um conjunto de favelas, definidas por uma geometria envoltória
- Executa ETL distribuído (calc_all(workers=N))
- Resolve clusters (“Grajaú”, “Butantã”, …)

### Favela

- Unidade de processamento
- Carrega períodos/épocas
- Converte LAZ/COPC → Arrow
- Executa transformações (calc_hag(), calc_vielas(), …)
- Salva camadas modulares, seja em arquivos físicos ou em nuvem (B3, S3, R2)

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

ou para desenvolvimento local:

```bash
pip install -e .
```

### Exemplo de Uso

#### Carregando uma favela e calculando HAG

```Python
from flaz import Favela
fl = Favela("São Remo").periodo(2017)
fl.calc_hag()
```

ou diretamente pela linha de comando:

```bash
flaz calc-hag --favela "São Remo" --ano 2017
```


#### Extraindo vielas

```bash
flaz calc-vielas --favela "Heliópolis" --force --ano [2017, 2020]
```

#### Processando um cluster inteiro (distribuído)

```bash
flaz calc-all --distrito "Grajaú" --force --use-cloud-run --destino "R2"
```

#### Processando API para uso no FVIz

```bash
flaz calc-more --ano 2017 --api ../fviz/apps/web/public/api
```

## Princípios da Biblioteca

1. OO Semântico

    Classes representam ideias urbanas reais, não apenas estruturas técnicas.

2. Transformações Puras

    `calc_*` nunca altera estado; sempre retorna um FViz completo.

3. Camadas Modulares

    Cada feature tem seu próprio arquivo.
    FLAZ é um data cube espacial-temporal, não um arquivo monolítico.

4. Runtime Leve

    `.arrow or .flaz` é otimizado para WebGL/WebGPU:
    >compacto, rápido e ideal para visualização.

## Desenvolvimento e Testes

Instalação em modo editável
```
pip install -e .
```

## Testes

`pytest --doctest-modules src/flaz`

## Versões e roadmap

#### TODO

## 🤝 Contribuições

Contribuições serão bem-vindas quando o projeto alcançar a versão 1.0.
Sinta-se livre para abrir issues, propor features e sugerir melhorias, copiar, modificar executar.

## Licença

A definir — provavelmente Apache 2.0 com notice.