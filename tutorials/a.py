def func(a, b=2, *args, **kwargs):
    # print(locals())

    if n := kwargs.get("clave", False):
        print(f"clave = {n}")

# func(5, 6, 7, 8, 9, p=4, gsad=412, clave=4)


class Plot:

    DEFAULTS = {
        "test": 2,
        "test_2": [2, 3, 4]
    }

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name

        self.DEFAULTS.update(kwargs)
        print(self.DEFAULTS)


Plot("asda", p=4, test=124)


rectangulos(1,5, 3,6, 2,7, 2,9)