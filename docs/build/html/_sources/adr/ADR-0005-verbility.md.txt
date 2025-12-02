# ADR-0005 — Adoção da gramática Verbility como sistema de habilidades do FLAZ

## Status
Accepted

## Contexto
O FLAZ cresceu rapidamente e passou a incorporar múltiplas “capacidades”:  
- HAG (Height Above Ground)  
- Vielas (extrair espaços vazios navegáveis)  
- MDT/MDS  
- FFTs espaciais  
- Sunibility / Windibility  
- Métricas multi-escala  
- Campos ambientais (hazard fields)  
- Entre outras…

Estas capacidades não são apenas “funções” — elas são **operadores de leitura
da cidade**, filtros fenomenológicos que revelam aspectos distintos da favela
3D+t.

O problema:  
Como organizar essas habilidades para que:

- sejam modulares,
- tenham API coerente,
- sejam descobertas automaticamente,
- reflitam significado semântico,
- e dialoguem naturalmente com o plugin system `_calc_*`?

A inspiração veio da linguística, da computação e do fenomenológico:  
cada cálculo é um **verbo**;  
cada estado resultante é uma **habilidade**;  
o FLAZ, portanto, tornou-se uma entidade com **verbilidade**.

Um mapa mental emergiu:

> **Verbo (operação) → Habilidade (estado derivado) → Layer persistido**

Ou seja:  
**hag() → HAGbility**  
**vielas() → Vielability**  
**mdt() → MDTbility**

De forma natural, intuitiva e expansionável.

## Decisão
O FLAZ adotará uma **gramática formal de habilidades**, chamada **Verbility**, que
define:

1. **Todo verbo computacional do FLAZ corresponde a um adjetivo de habilidade**,  
   formado pelo padrão:

`<VERBO>bility`


Exemplos:
- hag() → HAGbility  
- mdt() → MDTbility  
- vielas() → Vielability  
- sun() → Sunibility  
- wind() → Windibility

2. **Cada habilidade é representada como uma camada derivada**, registrada em:

`f.layers["<verbo>"]`


3. **A habilidade pertence ao domínio semântico da favela**, não ao domínio
computacional.  
Isto significa que `HAGbility` é um *estado interpretado*, não apenas uma
matriz derivada.

4. **A gramática complementa o plugin system**:
- `_calc_<verbo>()` produz a table Arrow
- `<verbo>()` executa a operação
- `<VERBO>bility` nomeia o estado resultante

5. **A Verbility organiza a documentação e a interface pública** do FLAZ.

6. **Habilidades com nomes muito próximos (ex.: MDTbility e MDSbility) são
consideradas “irmãs lexicais”** e serão documentadas juntas.

7. O FLAZ, portanto, passa a operar com uma **linguagem própria**, consistente e
declarativa.

## Alternativas consideradas

### 1. Dar nomes técnicos tradicionais (ex.: `compute_hag`, `derive_mdt`)
Desvantagens:
- API pesada e pouco intuitiva.
- Não comunica significados.
- Quebra a estética do FLAZ.

### 2. Criar classes explícitas para cada habilidade
Elegante, porém:
- Cerimônia excessiva,
- Muito boilerplate,
- Pouca fluidez.

### 3. Usar nomes soltos, não padronizados
Perderia coerência, dificultaria documentação e API pública.

### 4. Usar apenas números e códigos (como PDAL stages)
Rápido, mas totalmente antagônico à filosofia declarativa do projeto.

## Consequências

### Positivas
- API natural: `f.hag()`, `f.vielas()`, `f.sun()`
- A documentação pode organizar habilidades por domínios semânticos.
- @fviz pode mapear habilidades diretamente para botões / toggles / HUD.
- Permite uma escrita científica mais clara (os “-bilities” se tornam
conceitos).
- Facilita testes e extensões.
- A gramática reflete a ontologia do projeto.

### Negativas
- Requer disciplina para manter nomes consistentes.
- Pode exigir refinamento de nomes em casos ambíguos.
- A criatividade da gramática precisa ser preservada com cuidado.

## Notas finais
A Verbility é mais que uma convenção de nomes — é a **linguagem oficial do FLAZ**.

Ela estabelece uma ponte entre computação, semântica urbana e a estética do
projeto.  
Cada habilidade é uma forma de ver a favela.  
Cada verbo é um gesto interpretativo.

Com esta ADR, o FLAZ adota uma gramática viva, expansível e coerente.

A cidade ganha **habilidades**, e o FLAZ ganha voz.
