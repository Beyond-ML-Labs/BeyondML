from torch.nn import Module

class SelectorLayer(Module):

    def __init__(
        self,
        sel_index
    ):
        super().__init__()
        self.sel_index = sel_index

    @property
    def sel_index(self):
        return self._sel_index
    @sel_index.setter
    def sel_index(self, value):
        if not isinstance(value, int):
            raise TypeError('sel_index must be integer-valued')
        self._sel_index = value

    def forward(self, inputs):
        return inputs[self.sel_index]
