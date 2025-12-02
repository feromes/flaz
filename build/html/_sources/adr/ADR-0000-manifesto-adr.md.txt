# ADR-0000 — Adoção de Architecture Decision Records (ADRs)

## Status
Accepted

## Contexto
O projeto FLAZ nasceu como uma arquitetura viva, modular, experimental e
cientificamente rastreável.  
A cada semana surgiram novas decisões estruturantes: formatos de dados,
organização do ETL, padrões de módulos, convenções de nomes, filosofia de
persistência, plugin system, camadas do ETLVA, integração com o @fviz, políticas
de licença e governança.

Sem um registro claro e versionado dessas decisões, o risco é que:
- o racional se perca,
- decisões reapareçam como dúvidas no futuro,
- colaboradores não entendam o porquê das escolhas,
- a arquitetura fique incoerente ao longo do tempo,
- a documentação não comunique a filosofia do projeto.

Precisávamos de um mecanismo simples, textual, versionado e acessível que
capturasse **como pensamos** e não apenas **o que construímos**.

## Decisão
O FLAZ adotará **Architecture Decision Records (ADRs)** como mecanismo oficial
para documentar todas as decisões arquiteturais relevantes, seguindo o padrão:

- Arquivos individuais em Markdown
- Nomeados como `ADR-XXXX-titulo.md`
- Localizados em `docs/source/adr/`
- Referenciados automaticamente na documentação Sphinx
- Cada ADR contendo: *Contexto, Decisão, Alternativas, Consequências*

ADRs passam a ser parte formal do ciclo de vida do projeto e devem acompanhar
qualquer mudança estrutural relevante.

## Alternativas consideradas
- **Registrar decisões em issues do GitHub**  
  Disperso, pouco durável e difícil de rastrear.
- **Comentários em código**  
  Não capturam a meta-arquitetura, apenas detalhes locais.
- **Documento único de arquitetura**  
  Tende a ficar enorme, obsoleto e difícil de versionar granularmente.
- **Nenhuma forma de registro**  
  A arquitetura evolui sem memória — inviável para um projeto que depende de
  rastreabilidade científica e técnica.

## Consequências

### Positivas
- Decisões tornam-se **transparentes** e rastreáveis.
- Colaboradores entendem o “porquê” das escolhas.
- A arquitetura se mantém **coesa**, mesmo com múltiplos módulos e camadas
  (ETLVA, plugin system, .flaz, @fviz).
- A documentação do FLAZ passa a refletir a sua **filosofia**, não apenas sua API.
- Torna-se fácil revisar, reverter ou evoluir decisões com clareza.

### Negativas
- Exige disciplina de escrita e manutenção.
- Pode aumentar o overhead inicial ao criar novos módulos.
- Algumas decisões rápidas precisarão ser formalizadas.

## Notas finais
ADRs serão tratadas como **parte da própria arquitetura**, e não como um apêndice.
O ADR-0000 funciona como o manifesto do projeto: o FLAZ é uma entidade viva,
e sua evolução deve ser narrada, registrada e compreendida.

Esta decisão estabelece a base para todas as demais ADRs.
