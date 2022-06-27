import torch


class FilterLayer(torch.nn.Module):

    def __init__(
        self,
        is_on=True
    ):

        super().__init__()
        self.is_on = is_on

    @property
    def is_on(self):
        return self._is_on

    @is_on.setter
    def is_on(self, value):
        if not isinstance(value, bool):
            raise TypeError('is_on must be Boolean')
        self._is_on = value

    def forward(self, inputs):
        if self.is_on:
            return inputs
        else:
            return torch.zeros_like(inputs)

    def turn_on(self):
        self.is_on = True

    def turn_off(self):
        self.is_on = False
