from flaz import Favela

def test_favela_init():
    f = Favela("São Remo")
    assert f.nome == "São Remo"
