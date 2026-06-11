def test_package_imports() -> None:
    import multi_link_cartpole_rl

    assert multi_link_cartpole_rl.__version__ == "0.1.0"


def test_placeholder_environments_import() -> None:
    from multi_link_cartpole_rl.envs import MultiLinkCartPoleEnv, SingleLinkCartPoleEnv

    single_link_env = SingleLinkCartPoleEnv()
    multi_link_env = MultiLinkCartPoleEnv()

    assert single_link_env.config.num_links == 1
    assert multi_link_env.config.num_links == 2
