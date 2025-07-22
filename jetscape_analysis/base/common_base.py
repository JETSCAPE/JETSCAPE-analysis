"""Analysis base class.

.. codeauthor:: Mateusz Ploskon
"""


################################################################
class CommonBase:
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    # ---------------------------------------------------------------
    # Add an arbitrary attribute to the class
    # ---------------------------------------------------------------
    def set_attribute(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    # ---------------------------------------------------------------
    # Return formatted string of class members
    # ---------------------------------------------------------------
    def __str__(self) -> str:
        s = []
        variables = self.__dict__.keys()
        for v in variables:
            s.append(f"{v} = {self.__dict__[v]}")
        values = "\n .  ".join(s)
        return f"[i] {self.__class__.__name__} with \n .  {values}"
