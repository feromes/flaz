class FLaz:
    """
    Objeto fundamental do ecossistema FLAZ.

    A classe :class:`FLaz` representa a menor unidade coerente de dado espacial
    no sistema: um artefato geoespacial persistente que combina:

    - Geometria (2D, 3D ou 3D+t)
    - Metadados semânticos, provenientes de seu contexto territorial de origem (ex: favela, distrito)
    - Representação binária otimizada para leitura, recorte e visualização

    Conceitualmente, o :class:`FLaz` ocupa um papel análogo ao de um
    :class:`geopandas.GeoDataFrame`, porém com foco em:

    - Grandes volumes de dados
    - Representações multiescala
    - Fluxos ETL reprodutíveis
    - Visualização 3D+t
    - Análise morfológica urbana

    Um objeto :class:`FLaz` pode ser entendido como um **token espacial vivo*
    ele não é apenas um arquivo, mas uma entidade operável, versionável e
    atravessável por múltiplas camadas do sistema (ETL, visualização, análise,
    machine learning).

    A criação de um :class:`FLaz` pode ocorrer por leitura direta de um arquivo
    ou URL:

    >>> import flaz
    >>> #f = flaz.read("http://www.seila.com/fernando.flaz")

    Ou por construção manual a partir de geometria e seu contexto:

    >>> #f = FLaz(geom=geom, meta=favela.meta)

    Relação com outras classes:
    ----------------------------
    - :class:`Favela` é uma interpretação territorial original e semântica de um candidato à, ou containner de um ou mais
      objetos :class:`FLaz`.
    - :class:`Favelas` é um agregador de múltiplas instâncias :class:`Favela`.
    - :class:`FLaz` **não depende semanticamente de Favela**, mas pode ser usado
      por ela como fonte de dados brutos.

    Atributos principais:
    ---------------------
    geom : objeto geométrico
        Geometria associada ao artefato (polígono, pontos, voxel, etc.).

    meta : dict
        Metadados descritivos, técnicos e científicos do objeto.

    Exemplos de uso no ecossistema:
    -------------------------------
    >>> from flaz import Favela, Favelas
    >>> import flaz
    >>>
    >>> #f = flaz.read("http://www.seila.com/fernando.flaz")
    >>> #fl = Favela("São Remo").hag(force_recalc=True)
    >>> #fls_nn5 = fl.hag().nearest_neighbors(k=5)
    >>>
    >>> #for f in fls_nn5:
    # ...     print(f.nome, f.area_construida())
    >>>
    >>> #fls = Favelas(distritos=["Cidade Ademar", "Grajaú"])
    >>> #for f in fls:
    # ...     print(f.nome)

    Filosofia de design:
    --------------------
    - A classe :class:`FLaz` é deliberadamente minimalista.
    - Não carrega semântica urbana por padrão.
    - É orientada a **dados, não a narrativas**.
    - Toda semântica (favela, distrito, período, índices, métricas)
      emerge em camadas superiores.

    Ela pode ser **tudo ou nada**, mas nunca deve ser ambígua:
    ou representa um artefato espacial válido, ou falha de forma explícita.

    """

    FORMAT = "FLAZ"
    VERSION = "0.1.0"

    DEFAULT_CRS = "EPSG:4326"

    REQUIRED_META_FIELDS = {
        "nome",
        "origem",
        "data",
        "crs"
    }

    DEFAULT_DRIVER = "flaz.v1"

    STRICT = True

    def __init__(self, geom=None, meta=None, binfile=None):
        self.geom = geom
        self.meta = meta or {}
        self.binfile = binfile

    def nome(self):
        """
        Retorna o nome do objeto FLaz a partir dos metadados.
        """
        return self.meta.get("nome", "desconhecido")