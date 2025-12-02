# ADR-0002 — Adoção de Parquet/Arrow como formato persistido oficial do FLAZ

## Status
Accepted

## Contexto
O FLAZ precisava de um formato de persistência que atendesse simultaneamente:

- desempenho de leitura e escrita,
- interoperabilidade entre Python ↔ TypeScript,
- semântica colunar (ideal para nuvens de pontos e ETL),
- integração nativa com Arrow, PyArrow e engines de alto desempenho,
- compatibilidade com sistemas distribuídos (Ray, Cloud Run, OGDC),
- compressão eficiente,
- capacidade de particionamento inteligente (prefixos, ranges, Morton),
- capacidade de sobreviver ao tempo (durável e estável).

Além disso, o FLAZ precisava ser lido **diretamente no @fviz**, sem reprocessamento,
permitindo que dados complexos fossem transmitidos de forma simples pelo browser.

No núcleo do ETLVA, a etapa **Load** exigia um formato confiável que pudesse funcionar
como ponte entre o processamento pesado (Python) e a visualização leve e viva (JS/TS).

Formatos tradicionais como LAS/LAZ, COPC ou HDF5 trazem limitações:
- não são colunares,
- portabilidade limitada para o browser,
- difícil integração com Morton/Z-order,
- pouca compatibilidade nativa com Arrow,
- exigem engines específicas.

O FLAZ precisava de um formato universal, moderno, colunar e orientado a ciência de dados.

## Decisão
O formato oficial de persistência do FLAZ será **Parquet**, com representação interna
**Arrow-native**.

Isso significa que:

- **todo resultado persistido do FLAZ** (incluindo o token `.flaz`, XYZ, classificação, meta e layers) será salvo em `.parquet`;
- todas as tabelas internas do FLAZ serão representadas como **pyarrow.Table**;
- a comunicação com o @fviz será sempre via **Arrow → Parquet → Arrow JS** (automaticamente inferido pelo navegador);
- o particionamento e ordenação seguirão o prefixo Morton (Z-order) para eficiência de leitura;
- a etapa “Load” do ETLVA criará arquivos `.parquet` estáveis, versionáveis e transportáveis.

Essa decisão garante que o `.flaz` é um objeto vivo **independente de formato** e que sua
persistência é **colunar, rápida, comprimida e interoperável**.

## Alternativas consideradas

### COPC / LAZ
Vantagens:
- Otimizado para point clouds.

Desvantagens:
- Foco em LiDAR, mas não em atributos complexos do FLAZ.
- Não colunar.
- Fraco para comunicação com JS/TS.

### HDF5 / Zarr
Vantagens:
- Flexíveis.

Desvantagens:
- Ecosistema mais pesado.
- Integração fraca com Arrow.
- Não são ideais para streaming em browser.

### Arrow IPC / Feather
Vantagens:
- Leitura instantânea.

Desvantagens:
- Menos difundido que Parquet para pipelines grandes.
- Ferramental de compressão e particionamento mais limitado.

### GeoParquet
Vantagens:
- Focado em vetores.

Desvantagens:
- O FLAZ trabalha primariamente com *pontos*, não com geometria WKB.

## Consequências

### Positivas
- FLAZ passa a ser **nativamente interoperável** com Python, TypeScript, Spark, DuckDB, Polars, BigQuery, Snowflake e engines futuras.
- @fviz consegue consumir os dados diretamente com excelente performance.
- Arrow permite **zero-copy** entre camadas.
- Parquet permite **compressão eficiente**, partições por Morton, e ordenação colunar.
- O FLAZ se torna compatível com pipelines distribuídos (Ray/OGDC) sem adaptações.

### Negativas
- Pode ser necessário lidar com a semântica colunar durante certos cálculos.
- Algumas estruturas 3D mais ricas precisarão ser derivadas em memória.

## Notas finais
A escolha por Parquet/Arrow não é apenas pragmática — ela é **ontológica**.

Ela transforma o `.flaz` em uma entidade que respira entre linguagens, ambientes e escalas.

Essa decisão garante que o FLAZ seja um formato **portável, durável, distribuível** e
intrinsecamente conectado à ciência de dados moderna.

Esta ADR fundamenta o modo como o FLAZ fala com o mundo.
