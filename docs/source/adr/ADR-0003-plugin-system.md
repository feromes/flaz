# ADR-0003 — Sistema de Plugins `_calc_*` como mecanismo oficial de operações do FLAZ

## Status
Accepted

## Contexto
Ao longo do desenvolvimento, o FLAZ começou a acumular diversas operações complexas:

- HAG (Height Above Ground)
- MDS/MDT
- Vielas e espaços vazios
- Concavidades, envelopes, FFTs
- Métricas multi-escala
- Sunibility, windibility, shadowability
- ETC…

O núcleo `Flaz` não poderia se transformar em um monólito cheio de métodos pesados.
Era preciso criar uma arquitetura:

- **modular**,  
- **escalável**,  
- **descoberta automaticamente**,  
- **fácil de estender**,  
- **com operações isoladas para testes**,  
- **e que não sujasse a classe principal**.

Ao mesmo tempo, o FLAZ precisava manter uma **API elegante**, onde o usuário
pudesse escrever simplesmente:

```python
f = Flaz(...)
f.hag()
f.mds()
f.vielas()
f.flaz()
```

Ou seja: uma API verbosa, viva, natural — sem carregar complexidade técnica.

Para conciliar legibilidade com modularidade, o projeto precisava de um sistema
de verbos plugáveis, onde cada operação é um módulo autônomo.

### Decisão

O FLAZ adotará um plugin system baseado em funções _calc_*:

1. Cada operação pesada é implementada em um módulo separado:

```
flaz/ops/_calc_hag.py
flaz/ops/_calc_vielas.py
flaz/ops/_calc_fft.py
...
```

2. Cada módulo define uma única função com a assinatura:

```Python
def _calc_<verbo>(flaz: Flaz, *args, **kwargs) -> pyarrow.Table:
    ...
```

3. O núcleo Flaz irá, durante a inicialização, varrer automaticamente o
pacote flaz/ops/, buscar todas as funções _calc_* e:

- registrar o verbo correspondente,
- injetar um método público na instância,
- conectar operação → layer → persistência → metadata.

4. O método público resultante será do tipo:

`f.<verbo>()`


5. O layer resultante será registrado automaticamente em:

`f.layers["<verbo>"] = tabela_arrow`


6. O método _register_layer continuará sendo o responsável por registrar os
artefatos derivados e manter rastreabilidade.

### Em suma:

> O FLAZ aprende novas habilidades simplesmente colocando arquivos no diretório de operações.

Sem tocar no core.

### Alternativas consideradas
1. Colocar todas as operações na classe Flaz

Desvantagens:

- polui a API,
- aumenta o acoplamento,
- dificulta testes,
- inviabiliza evolução modular.

2. Criar subclasses para cada habilidade (HAGbility, Vielability…)

Melhor semântica, mas ainda gera acoplamento e não cria um pipeline declarativo limpo.

3. Registrar manualmente cada verbo

Quebra a promessa de extensibilidade automática.

4. Criar uma estrutura de “command objects”

Apesar de elegante, seria excesso de cerimônia para o propósito.

### Consequências
#### Positivas

- O núcleo do FLAZ permanece pequeno, estável e elegante.
- Cada operação é testável de forma isolada.
- O sistema é naturalmente extensível (basta adicionar um - arquivo).
- A API do usuário é intuitiva (f.hag(), f.vielas(), etc.).
- Facilita documentação automática no Sphinx.
- Permite criar “habilidades“ (Verbility) de maneira orgânica.

#### Negativas

- Exige disciplina de nomenclatura (_calc_*).
- Pode gerar muitos módulos no diretório de operações (esperado - em um projeto vivo).
- Requer lógica de introspecção bem definida no núcleo.

### Notas finais

Este plugin system faz parte da identidade do FLAZ.
Ele transforma um pipeline rígido em uma linguagem viva, onde novas
operações nascem por adição, não por modificação.

O FLAZ se torna um organismo: o núcleo é pequeno, e as habilidades crescem
nas bordas.

Esta ADR estabelece o mecanismo que permitirá que o projeto viva por anos.
