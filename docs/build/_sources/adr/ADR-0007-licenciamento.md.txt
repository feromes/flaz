# ADR-0007 — Licenciamento e Governança do Ecossistema FLAZ/FVIZ/FavelaViz

## Status  
Accepted

## Contexto  
O ecossistema FLAZ abrange três camadas distintas — **bibliotecas**, **infraestrutura**, e **aplicação final** — cada uma com públicos, responsabilidades e riscos diferentes.

1. **flaz (Python)**  
   - biblioteca científica e operacional para ETL 3D+t  
   - usada por pesquisadores, desenvolvedores, pipelines e scripts  
   - precisa ser auditável, estável, interoperável e amplamente adotável  

2. **fviz (TypeScript/React)**  
   - biblioteca cliente para leitura e visualização do `.flaz` no navegador  
   - usada por aplicações web, dashboards e experimentos  
   - deve ser fácil de integrar, extender e versionar  

3. **FavelaViz (aplicação final em Next.js)**  
   - produto consolidado de alto valor, com marca, UI, storytelling, HUD, camadas premium  
   - envolve governança institucional, branding e acordos de titularidade  
   - é a interface pública e pedagógica que comunica o projeto para o mundo  

Além disso, existe a quarta camada **latente/futura**:

4. **OGDC / Open GeoDataCube**  
   - infraestrutura distribuída para ingestão, versionamento e processamento massivo  
   - não faz parte do MVP atual, mas influencia decisões de licenciamento

O desafio:  
**Escolher um modelo que permita colaboração científica global (para libs), mas proteja o produto final (aplicação FavelaViz) e suas narrativas.**

---

## Decisão  
Foi adotado um modelo de **licenciamento dual**, dividido por camadas, alinhado com práticas contemporâneas de projetos open-core.

### 1. Bibliotecas fundamentais  
**flaz (Python)** → **Apache 2.0**  
**fviz (TypeScript)** → **Apache 2.0**

Justificativas:

- permite uso comercial, acadêmico e governamental sem fricção;  
- garante proteção contra “patent trolling”;  
- incentiva adoção;  
- alinha com padrões de ecossistemas como Arrow, Parquet, PDAL, GeoPandas, MapLibre;  
- perfeito para bibliotecas científicas.

### 2. Aplicação final  
**FavelaViz (Next.js)** → **Licença Dual (Apache 2.0 + Licença Específica / Proprietária)**

O código público (ex.: exemplos, páginas de demo, HUD básico) pode ser Apache 2.0.  
Mas o produto completo, incluindo:

- layout final,  
- temas,  
- storytelling,  
- design premium,  
- integração com datasets oficiais,  
- marca FavelaViz,  
- dashboards e roteiros pedagógicos,  
- módulos interpretativos e animações específicas,  

permanece sob **licença proprietária controlada por consórcio**.

Isso permite:

- abrir a tecnologia, mas proteger o produto;  
- evitar forks de terceiros com a mesma identidade;  
- assegurar consistência institucional;  
- controlar governança e narrativa pública;  
- viabilizar parcerias e financiamentos.

### 3. Branding e titularidade  
O nome **FLAZ** e **FavelaViz** são **marcas** e pertencem ao consórcio (instituições envolvidas + autor(es)).  
Código → licenciado  
Marca → protegida  

Isso permite que o ecossistema seja aberto, mas o “guarda-chuva cultural” fique preservado.

---

## Alternativas consideradas

### 1. Tudo MIT  
Muito permissivo; permitiria cópias comerciais da aplicação final sem restrições.

### 2. Tudo Apache  
Protegeria menos a narrativa do FavelaViz (marca, UI premium, curadoria).

### 3. Tudo proprietário  
Contrário à filosofia do projeto, limitaria adoção acadêmica e colaboração.

### 4. GPL / AGPL  
Criaria atrito para integração com infra moderna (Next.js, React, servidores escaláveis).

---

## Consequências

### Positivas  
- bibliotecas abertas e fáceis de usar;  
- ecossistema tende a crescer e se difundir;  
- aplicação final protegida como produto;  
- governança clara entre autoria, titularidade e uso;  
- compatibilidade com FAPESP, universidades e futuros parceiros;  
- evita fragmentação e forks que prejudiquem a identidade do projeto.

### Negativas  
- exige comunicação clara na documentação;  
- necessidade de um arquivo `LICENSE` por repositório;  
- possível necessidade de um documento oficial de governança.

---

## Notas finais  
Este modelo dual combina o melhor de dois mundos:

- **inovação aberta e expansível** → flaz & fviz (Apache 2.0)  
- **produto institucional com curadoria e narrativa própria** → FavelaViz (licença dual)

A filosofia é simples:

> **A tecnologia é livre, a história é nossa.**

O FLAZ se estabelece assim como um ecossistema eticamente aberto, tecnicamente rigoroso e culturalmente protegido — honrando a ciência, a cidade e a estética do projeto.
