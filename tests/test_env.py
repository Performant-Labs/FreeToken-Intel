from freetoken.env import ENV


def test_env_defaults():
    assert ENV.SHELL_MAX_TOKENS.value == 2048
    assert ENV.USE_XMX.value is True
