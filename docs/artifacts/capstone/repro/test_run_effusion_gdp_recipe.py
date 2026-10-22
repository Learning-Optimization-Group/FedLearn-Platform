import pytest

import run_effusion_gdp_recipe as gdp


@pytest.mark.parametrize('epsilon', gdp.EPSILONS)
def test_rdp_equivalent_budget_makes_the_engine_use_the_gdp_noise(epsilon):
    from fedlearn.privacy.dp_accountant import required_noise_multiplier
    cal = gdp.calibration(epsilon)
    engine_z = required_noise_multiplier(cal['rdpEquivalentEpsilon'], 1.0, gdp.federated.ROUNDS, gdp.DELTA)
    assert gdp.check_calibration(cal, engine_z) <= epsilon + 1e-6
    assert cal['gdpNoiseMultiplier'] < required_noise_multiplier(epsilon, 1.0, gdp.federated.ROUNDS, gdp.DELTA)


def test_a_run_with_the_wrong_noise_is_refused():
    cal = gdp.calibration(4.0)
    with pytest.raises(gdp.Refusal, match='noise_multiplier_mismatch'):
        gdp.check_calibration(cal, cal['gdpNoiseMultiplier'] * 1.001)
