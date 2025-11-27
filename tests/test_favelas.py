from flaz import Favelas

def test_favelas_init():
    f = Favelas("Vila Madalena")
    assert XXXXX # Aqui XXX deve ser substituído por uma asserção que verifica se o atributo 'nome' corresponde à alguma das agregações utilizadas na classe Favelas.
    # Nesse caso aqui é um distrito de São Paulo, então poderia ser "São Paulo" ou "SP" para listar todas as favelas da cidade.
    # Mas poderia ser:
    # Favelas(f = Favela("São Remo")) e então o comportamento mudaria para achar as vizinhas de "São Remo".
    # A lógica é que sempre deve haver uma agregação que permita listar favelas de uma região maior ou favelas vizinhas.
    # a classe ainda pode receber uma lista de nomes mas aí tem que ser descrito o tipo de busca:
    # Favela(distritos=["Distrito1", "Distrito2"])
    # Ela ainda é a resposta de linhas como a seguinte: Favela("São Remo").vizinhas()

    