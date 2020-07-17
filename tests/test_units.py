import pytest
import enzo


def test_init_neuron():
    n = enzo.units.Neuron()
    assert isinstance(n, enzo.units.Neuron)
    assert n.weights is None
    assert n.bias is None
    assert n.activation is enzo.activation.noactivation


def test_neuron_process_given_weights():
    inp = [0.2, 99, -5, 12]
    n = enzo.units.Neuron([[4.1], [0], [-2], [5]], bias=4)
    assert n.process(inp) == pytest.approx(74.82)


def patched_randn(*args):
    if args == (4, 1):
        return [[0.4124], [-0.1], [0.99131], [-0.757]]  # weights
    if args == ():
        return 4   # bias
    raise ValueError('unexpected args passed')


def test_neuron_process_without_weights(mocker):
    patched = mocker.patch('enzo.units.np.random.randn',
                           side_effect=patched_randn)
    inp = [0.2, 99, -5, 12]
    n = enzo.units.Neuron()
    assert n.process(inp) == pytest.approx(-19.85807)
    assert patched.call_args_list == [((4, 1),), ()]
