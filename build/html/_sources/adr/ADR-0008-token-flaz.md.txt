# ADR-0008 — O Formato `.flaz` como Token Espacial Vivo (128 bits)

## Status  
Accepted

## Contexto  
O FLAZ não nasceu como “mais um arquivo geoespacial”.  
Ele nasceu como **uma nova forma de pensar o espaço-tempo urbano**, principalmente no contexto de nuvens de pontos LiDAR, análises 3D+t e pipelines distribuídos.

Os formatos tradicionais — LAZ, COPC, Parquet, Arrow, GeoParquet — cumprem papéis essenciais, mas nenhum deles:

- incorpora **hierarquia espacial intrínseca**;  
- combina **dimensões 3D + tempo** em um único átomo;  
- representa o espaço como um **endereço codificável, navegável, paginável**;  
- permite **ranges** contínuos para armazenamento distribuído;  
- unifica **ETL → visualização → análise**;  
- funciona como chave primária universal para cidades 3D+t.

A solução emergiu naturalmente:  
**o `.flaz` não seria apenas um arquivo — seria um *token espacial vivo*.**

Um identificador compacto, determinístico, hierárquico e temporal que carrega dentro de si a “gramática do lugar”.

---

## Decisão  
O formato `.flaz` é definido como um **token de 128 bits**, composto por:

- **96 bits espaciais** (Morton 3D)  
- **32 bits temporais** (epoch, frame ou timestamp normalizado)  

Esse token é:

- ordenável  
- hierárquico  
- comparável  
- cruzável entre favelas, cidades e épocas  
- adequado para particionamento distribuído  
- transportável entre linguagens (Python, JS/TS, Rust, SQL)  
- capaz de ser indexado, filtrado, agrupado e animado

### Estrutura  
```
[ MortonX | MortonY | MortonZ | Time ]
```

Ao invés de armazenar “coordenadas e atributos”, o `.flaz` armazena **unidades vivas** de espaço-tempo.

---

## Propriedades-chave

### 1. **Hierarquia espacial fractal**  
Mais bits → maior resolução  
Menos bits → agregações naturais

Isso permite:

- pirâmides nativas  
- downsampling sem perda semântica  
- vizinhanças baratas  
- ranges espaciais contínuos  

### 2. **Temporalidade integrada**  
O token já sabe:

- a época (2017/2020/2024)  
- a ordem dos frames  
- o momento relativo de aquisição

Isso é essencial para:

- multitemporalidade do FavelaViz  
- animações 3D+t  
- reconstruções espaço-temporais  
- alinhamento de dados LiDAR com RGB/MDS

### 3. **Determinismo absoluto**  
Dado um ponto XYZ+t → sempre gera o mesmo token  
Dado um token → sempre recupera seu endereço

Isso torna o `.flaz`:

- replicável,  
- versionável,  
- auditável.

### 4. **Distribuição nativa (OGDC-ready)**  
Ranges de tokens permitem:

- particionamento horizontal  
- shards temporais  
- workers paralelos  
- processamento por tiles virtuais  
- compatibilidade com Cloud Run, Ray e OGDC futuro

### 5. **Interoperabilidade**  
O token `.flaz` viaja perfeitamente entre:

- Python (ETL)  
- TypeScript (visualização)  
- SQL/PostGIS (indexação)  
- Zarr/Arrow/Parquet (storage)

---

## Alternativas consideradas

### 1. Usar apenas Morton 2D  
Limitaria o projeto; perderíamos profundidade vertical.

### 2. Usar H3/S2  
Excelentes para 2D, mas não para 3D+t; exigiriam hacks complexos para Z e tempo.

### 3. Usar GUID/UUID  
Perderíamos completude semântica e ordenação espacial.

### 4. Usar Parquet “cru” como formato final  
Parquet é armazenamento, não ontologia.  
Faltaria a “alma fractal” do FLAZ.

### 5. Usar COPC como base  
COPC é perfeito para *storage* de nuvens brutas, mas não para semânticas, tokens, layers, animações.

---

## Consequências

### Positivas  
- padronização conceitual do FLAZ como linguagem  
- tokens unificados em todo o ecossistema  
- pipeline ETLVA simplificado  
- animações e ranges espaciais triviais  
- compatibilidade direta com OGDC  
- fácil de reproduzir cientificamente  
- reduz acoplamento entre camadas (ETL/Viewer/Storage)  
- compressão semântica poderosa (128 bits carregam tudo)

### Negativas  
- exige implementação consistente em Python e TypeScript  
- precisa de testes rigorosos de bitwise e decodificação  
- documentação precisa detalhar invariantes do token  

---

## Notas finais  
O `.flaz` não é apenas um “arquivo”.

Ele é a **unidade mínima da geografia operacional do projeto**,  
um átomo hierárquico que carrega:

- espaço  
- tempo  
- vizinhança  
- escala  
- resoluções múltiplas  
- navegabilidade  
- consistência universal  

É o tijolo conceitual que permite que:

> **o FLAZ seja não um formato, mas uma linguagem viva do espaço-tempo.**

A cidade não é salva como dados —  
é salva como *geometria simbólica*.

