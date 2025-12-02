# ADR-0010 — Chunking Hierárquico do Formato .FLAZ (512 KB + Morton Z-Index)

**Status:** Proposta  
**Data:** 2025-12-02  
**Autores:** Fernando Gomes & B (ChatGPT)

---

## 1. Contexto

O FLAZ está sendo definido como um **formato hierárquico, tokenizado e espacial**, baseado em Morton 3D/4D de 128 bits.  
Para suportar **carregamento progressivo, streaming eficiente e experiência visual “viva”**, precisamos de um mecanismo de particionamento físico dos dados em **chunks pequenos**, previsíveis, paralelizáveis e servidos diretamente por storage distribuído (B2, R2, S3, Cloudflare Worker URLs, etc.).

Sem chunking:

- um único `.flaz` pode ficar grande demais;
- o carregamento inicial pode gerar *lag* ou travamentos perceptíveis;
- o visualizador perde a sensação de “adensamento hierárquico” ao carregar;
- não há granularidade para caching, prefetching ou priorização espacial.

Ao mesmo tempo, o **Morton Z-index** já organiza o espaço numa ordem que preserva localidade, ideal para ordenação parcial e chunking.

---

## 2. Decisão

O arquivo `.flaz` **será armazenado como múltiplos chunks independentes**, cada um com:

- **Tamanho máximo alvo:** **512 KB**  
- **Ordem de armazenamento:** crescente no **Morton Z-index**
- **Conteúdo do chunk:**  
  - pontos (XYZ + atributos essenciais),  
  - metadados locais (bounding box, min/max, estatísticas),  
  - referência ao token espacial correspondente.
- **Arquivo índice (`.flaz.index`)** contendo:  
  - lista dos chunks,  
  - faixa de Morton codes coberta por cada chunk,  
  - tamanho real, hash e metadados,  
  - níveis hierárquicos implícitos.

### Motivações

1. **Experiência visual hierárquica (“densidade emergente”)**  
   Carregar os chunks em ordem Morton cria a sensação de que a nuvem vai **aparecendo**, **adensando** e **refinando**, como um zoom fractal dentro da favela.

2. **Streaming suave e responsivo**  
   Chunks de 512 KB equilibram latência, paralelismo e cache (Cloudflare/B2), permitindo exibição imediata de pontos parciais.

3. **Processamento distribuído**  
   Pipelines Ray/Rust/Go podem processar chunks de forma independente.

4. **Custo computacional previsível**  
   512 KB está alinhado a limites ótimos de CDN, latência e decompressão (ZSTD/LZ4).

---

## 3. Alternativas Consideradas

### A. Chunks maiores (1–4 MB)
- **Prós:** menos arquivos, menos metadata.  
- **Contras:** latência maior por chunk, pior progressividade.  
**Rejeitado.**

### B. Chunks menores (64–256 KB)
- **Prós:** maior progressividade.  
- **Contras:** requests demais, overhead no CDN e no loader.  
**Rejeitado.**

### C. Divisão por grids fixos (bounding boxes regulares)
- **Prós:** simples.  
- **Contras:** fragmenta estruturas naturais, não mantém localidade como Morton.  
**Rejeitado.**

### D. Arquivo monolítico
- **Prós:** implementação mais simples.  
- **Contras:** péssimo para Web 3D+t, sem streaming granular.  
**Rejeitado.**

---

## 4. Consequências

### Positivas
- Visualização orgânica e progressiva no FavelaVIZ.  
- Redução significativa do **Time-to-First-Points (TTFP)**.  
- Excelente integração com cache e prefetching.  
- Suporte natural a LOD baseado em Morton.  
- Paralelismo perfeito no ETLVA.

### Negativas
- Mais arquivos para gerenciar no storage.  
- Requer um indexador robusto no backend.  
- Viewer precisa de scheduler de chunks (já planejado).

---

## 5. Implementação (primeira versão)

### ETL FLAZ (Python)

1. Ler Arrow table  
2. Ordenar por Morton 128-bit  
3. Estimar quantos pontos ≈ 512 KB  
4. Criar chunks sequenciais  
5. Salvar como:  

```Python
/favelas/sao_remo/2024/flaz/chunk_00001.flz
/favelas/sao_remo/2024/flaz/chunk_00002.flz
...
```

6. Gerar `flaz.index` (Arrow/JSON/Zarr) com:  
- `morton_min`, `morton_max`  
- `bbox` local  
- `hash`, `size`  
- posição sequencial

### Loader @fviz (TypeScript)

1. Carrega `flaz.index`  
2. Ordena os chunks por `morton_min`  
3. Carrega progressivamente  
4. Renderiza cada chunk assim que decodificado  
5. Atualiza bounding box global e LOD  

---

**Conclusão:**  
Este ADR estabelece a base arquitetural para que o FLAZ tenha carregamento **hierárquico, fractal e progressivo**, totalmente coerente com a filosofia ETLVA e com a experiência 3D+t pretendida no FavelaVIZ.

