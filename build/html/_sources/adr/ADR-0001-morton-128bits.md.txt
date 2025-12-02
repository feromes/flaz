# ADR-0001 — Adoção de Morton (Z-Order) como estrutura hierárquica do formato `.flaz`

## Status
Accepted

## Contexto
O formato `.flaz` precisava de uma estrutura fundamental que fosse:

- **hierárquica** (multiescala),
- **espacialmente coerente** (navegável por vizinhança),
- **binária e compacta** (compatível com GPU),
- **ordenável** (para Parquet/Arrow),
- **estável no tempo** (para 3D+t),
- **nativa para indexação distribuída** (OGDC, Ray, Cloud Run),
- **com interoperabilidade conceitual** com octrees, tiles e voxelizações.

O problema central:  
**como representar espaço (e tempo) como tokens discretos, comparáveis e
consultáveis, sem depender de estruturas externas (KD-Tree, Octree, BVH)
durante a leitura?**

O projeto precisava de uma “geometria simbólica” que funcionasse como:
- endereço,
- índice,
- hierarquia,
- vizinhança,
- e identidade.

Além disso, como o FLAZ é uma arquitetura **de tempo**, o token precisava
encaixar naturalmente o 4D (XYZ + T) sem quebrar coerência.

## Decisão
O formato `.flaz` adotará **Morton codes (Z-order)** como estrutura central de
indexação e identidade espacial, com **128 bits**, distribuídos como:

- **96 bits espaciais** (interleaving de X, Y, Z)
- **32 bits temporais** (época, instante, camada)

A geração do token `.flaz` será:

```flaz_id = morton3D_96bits(x, y, z) interleaved with time_32bits```

E o ETL do FLAZ passa a produzir e persistir sempre essa coluna.

### Propriedades importantes:
- Ordenação lexicográfica coincide com proximidade espacial.
- Compatível com Parquet/Arrow: **sort → groupby → chunking** nativos.
- Navegação local é possível apenas comparando prefixos de bits.
- Permite “zoom” fractal sem estruturas auxiliares.
- Conecta naturalmente 3D+t com octrees, tiles e operações por escala.

## Alternativas consideradas

### 1. **Hilbert Curve**
Vantagens:
- Melhor localidade espacial (menos “saltos”).

Desvantagens:
- Computação mais pesada (CPU).
- Implementação complexa em 3D.
- Difícil integração em GPU e bibliotecas de alto desempenho.

### 2. **S2 / H3**
Vantagens:
- Robustos e maduros.

Desvantagens:
- Complexidade desnecessária para LiDAR/3D.
- Modelos esféricos (não cartesianos).
- Tempo como quarto eixo não se integra naturalmente.

### 3. **Octree explícita**
Vantagens:
- Estrutura intuitiva para 3D.

Desvantagens:
- Precisa ser reconstruída a cada ingestão.
- Armazena metadados demais.
- Difícil de serializar de forma colunar.

### 4. **KD-Tree / BVH**
Desvantagens gerais:
- Estruturas pesadas.
- Não são hierarquias simbólicas (não viram “tokens”).
- Incompatíveis com persistência em Parquet sem index externo.

## Consequências

### Positivas
- `.flaz` se torna um **átomo relacional do espaço-tempo**.
- Vizinhança, escala e tempo tornam-se **operações sobre bits**.
- A GPU pode trabalhar com tokens diretamente sem reconstruir árvores.
- Ray e pipelines distribuídos podem particionar o espaço por ranges.
- @fviz pode navegar o dataset como um fractal: zoom, cortar, agregar.
- A escrita científica do FLAZ ganha uma “fundação geométrica”.

### Negativas
- Morton tem pior localidade que Hilbert, embora suficiente para 99% dos casos.
- Requer normalização dos ranges XYZ antes da interleaving.
- O token precisa ser cuidadosamente documentado para evitar ambiguidade.

## Notas finais
A escolha do Morton não é técnica apenas — é **epistemológica**.  
O `.flaz` se torna uma *linguagem viva do espaço-tempo*, onde cada ponto é um
símbolo que carrega sua própria vizinhança, escala e tempo.

Essa decisão define o coração do FLAZ.
