# ADR-0004 — Adoção do modelo ETLVA como espinha dorsal conceitual do FLAZ

## Status
Accepted

## Contexto
O FLAZ rapidamente deixou de ser “apenas uma biblioteca” e tornou-se um **sistema
de vida dos dados**. Durante o desenvolvimento, ficou evidente que o fluxo clássico
de dados (Extract–Transform–Load) não era suficiente para expressar:

- a dimensão temporal do projeto (3D+t),
- a presença simultânea de múltiplas escalas (viela → favela → cidade),
- os ciclos de visualização, animação e narrativa,
- a relação interativa com o @fviz (frontend),
- a necessidade de rastreabilidade científica,
- o caráter declarativo de suas operações,
- a arquitetura fractal e modular das camadas.

Era necessário um modelo **mais amplo, narrativo e temporal**, capaz de integrar:

- processamento pesado (Python),
- visualização leve (JS/TS),
- animação,
- versionamento,
- estados intermediários,
- ontologia espacial,
- e um pipeline que se comporta como um organismo.

Assim surgiu o **ETLVA** — um acrônimo criado especialmente para o FLAZ:

> **Extract → Transform → Load → Visualize → Animate**

Ele não é apenas um fluxo; é um **modelo mental** e uma **filosofia de projeto**.

## Decisão
O FLAZ adotará **ETLVA como modelo conceitual e operativo oficial**.  
Cada camada, módulo e operação do sistema deverá:

1. **Extrair (Extract)**  
   Dados brutos de LiDAR, rasters, geometrias, tabelas e metadados.

2. **Transformar (Transform)**  
   Processar e produzir significado — normalizações, Morton, HAG, FFT, vielas,
   classificações, árvores, campos ambientais, etc.

3. **Carregar (Load)**  
   Persistir tudo em Parquet/Arrow e registrar como estado formal, versionável.

4. **Visualizar (Visualize)**  
   Disponibilizar camadas e tokens para o @fviz, Next.js e interfaces declarativas.

5. **Animar (Animate)**  
   Permitir a leitura temporal (3D+t), diffs, transições, sequências e histórias
   visuais — não apenas resultados estáticos.

Essa decisão implica:

- Cada módulo do FLAZ deverá declarar **em qual etapa do ETLVA trabalha**.
- A documentação técnica mapeará os componentes ao fluxo ETLVA.
- O @fviz será o ambiente oficial para **V** e **A**.
- O `.flaz` será o átomo que atravessa todas as etapas.

O ETLVA será usado **desde o código até a documentação**, como parte da identidade
do projeto.

## Alternativas consideradas

### ETL clássico
Insuficiente para representar:
- a dimensão temporal,
- a visualização interativa,
- o pipeline vivo do FavelaVIZ,
- a ideia de "estado narrativo" dos dados.

### ELT (Extract–Load–Transform)
Útil em pipelines de big data, mas não captura a semântica do FLAZ.

### Pipelines ad hoc
Rápidos, porém sem unificação conceitual e difíceis de manter.

### DAGs de orquestração (Airflow, Prefect…)
Poderiam organizar tarefas, mas não oferecem um **modelo epistemológico** que
integre computação e visualidade.

## Consequências

### Positivas
- O FLAZ tem agora uma **filosofia clara**.
- O pipeline é compreensível para humanos e máquinas.
- Cada módulo sabe seu lugar na linha do tempo dos dados.
- Visualização deixa de ser um apêndice; é uma **parte formal do processo**.
- Animação e temporalidade tornam-se nativas (não hacks adicionais).
- @fviz torna-se o espelho natural do backend.
- Documentação, código e arquitetura falam a mesma língua.

### Negativas
- Exige disciplina conceitual (é mais que pipeline: é narrativa).
- Obriga o código a manter coerência entre etapas.
- Requer documentação detalhada sobre o fluxo.

## Notas finais
O ETLVA é o **princípio vital** do FLAZ.  
Ele situa o projeto dentro de uma visão de dados como entidades vivas,
atravessadas por tempo, escala, visualidade e movimento.

É o modelo que permite ao FLAZ ser não apenas um pipeline técnico,
mas uma **estrutura estética, ética e espacial**.

Com esta ADR, o FLAZ assume oficialmente sua natureza temporal e visual.
