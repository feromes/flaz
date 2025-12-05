# ADR-0006 — Adoção da Arquitetura Declarativa (Capabilities → Operations → Layers)

## Status  
Accepted

## Contexto  
Durante a evolução do FLAZ, ficou evidente que o projeto não poderia ser organizado apenas como um conjunto de funções soltas ou métodos dentro de uma única classe.  
O FLAZ opera em múltiplas escalas, temporalidades e domínios:

- extração (Extract)
- transformação (Transform)
- persistência (Load)
- visualização (Visualize)
- animação (Animate)

Com isso surgiram inúmeras operações — HAG, MDT/MDS, Vielas, FFT, Sunibility, Windibility, Hazard Fields, embeddings, etc.  
Era necessário um modelo estrutural simples e expansível que:

- organizasse semanticamente o que o FLAZ pode fazer,
- isolasse operações,
- mantivesse o núcleo pequeno,
- preservasse coerência,
- fosse compatível com ETLVA,
- estruturasse a documentação e a API.

Daí emergiu a arquitetura declarativa:

**Capabilities → Operations → Layers**

Ela organiza não só o código, mas a própria ontologia do FLAZ.

---

## Decisão  
O FLAZ adotará a seguinte arquitetura declarativa triádica:

---

### 1. **Capabilities**  
São as **possibilidades semânticas** do FLAZ — o que ele é capaz de fazer.  
Exemplos:

- "Calcular HAG é uma capability."
- "Extrair vielas é uma capability."
- "Gerar FFT espacial é uma capability."

Capabilities representam **o que é possível**, não o como fazer.

---

### 2. **Operations**  
São as **implementações concretas** das capacidades.  
Cada operação é um módulo isolado com uma função `_calc_*`.

Exemplos:

```
_calc_hag()
_calc_vielas()
_calc_fft()
```

Cada operação:

- vive em `flaz/ops/`
- é descoberta automaticamente por introspecção
- é testável isoladamente
- é mapeada para um verbo público (`f.hag()`, `f.vielas()`…)

Operations representam **como fazer**.

---

### 3. **Layers**  
São os **resultados derivados** produzidos pelas operações.  
Cada operação `_calc_*` gera uma Arrow Table registrada como:

```
f.layers["hag"]
f.layers["vielas"]
f.layers["fft"]
```

Cada layer é:

- um estado derivado
- persistido em Parquet
- versionável
- consumido pelo @fviz

Layers representam **o que foi produzido**.

---

### Como a tríade opera na prática

1. *Capability:* “O FLAZ pode calcular HAG.”  
2. *Operation:* `_calc_hag()` é executada.  
3. *Layer:* `HAGbility` é registrada como resultado.

Este modelo organiza todo o FLAZ.

---

## Alternativas consideradas

### 1. Classe monolítica com todos os métodos  
- difícil de manter  
- acoplamento alto  
- impróprio para escalabilidade

### 2. Heranças profundas (HAGClass, VielasClass…)  
- burocrático  
- excesso de cerimônia  
- pouca fluidez

### 3. Organização apenas por verbos  
- capabilities ficariam implícitas  
- dificulta documentação

### 4. Organização apenas por layers  
- perderia clareza de produção e significado

---

## Consequências

### Positivas  
- Arquitetura escalável e elegante  
- Fácil adicionar novas capacidades (criar um `_calc_*`)  
- API pública coerente: `f.hag()`, `f.vielas()`  
- Organização clara para documentação  
- Layers tornam-se entidades científicas  
- @fviz opera diretamente sobre layers  
- Capabilities estruturam a ontologia do projeto  

### Negativas  
- Requer disciplina de separação conceitual  
- Eventuais renomeações para coerência  
- Documentação precisa acompanhar a evolução  

---

## Notas finais  
A arquitetura declarativa é a **espinha dorsal** do FLAZ.  
Ela separa claramente:

- **o que é possível** → *Capabilities*  
- **o que é feito** → *Operations*  
- **o que é produzido** → *Layers*  

Transformando o FLAZ em um sistema modular, interpretável e vivo.  
É a partir dessa tríade que o FLAZ evolui como linguagem, como pipeline e como organismo técnico e fenomenológico.
