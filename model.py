from torch import nn, from_numpy
import torch
import copy


class SimpleForward(nn.Module):
    def __init__(self, in_channels, out_classes, bias=True):
        super(SimpleForward, self).__init__()
        self.forward_1 = nn.Linear(in_channels, 24, bias=bias)
        #self.forward_2 = nn.Linear(24, 24, bias=bias)
        self.out = nn.Linear(24, out_classes, bias=bias)
        self.eval()

    def forward(self, x):
        x = nn.functional.relu(self.forward_1(x))
        #x = nn.functional.relu(self.forward_2(x))
        return nn.functional.softmax(self.out(x), dim=-1).argmax(dim=-1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self

    def mutate(self, mutation_rate, deepcopy=False):
        child = copy.deepcopy(self) if deepcopy else self
        for param in child.parameters():
            mask = torch.rand_like(param.data) < mutation_rate
            mutation = torch.randn_like(param.data) * mask
            param.data += mutation
        return child

    def crossover(self, other, bias=0.5, deepcopy=False):
        child = copy.deepcopy(self) if deepcopy else self
        for child_param, other_param in zip(child.parameters(), other.parameters()):
            mask = torch.rand_like(child_param.data) < bias
            child_param.data = torch.where(mask, child_param.data, other_param.data)

        return child


def load_model(path):
    state_dict = torch.load(path)
    input_size = state_dict["forward_1.weight"].shape[1]
    output_size = state_dict["out.weight"].shape[0]
    use_bias = "forward_1.bias" in state_dict
    model = SimpleForward(input_size, output_size, use_bias)
    return model.load(path)
