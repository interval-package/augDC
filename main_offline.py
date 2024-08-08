from main import main_offline_train, get_config_off

if __name__ == "__main__":
    args, env, kwargs = get_config_off()
    main_offline_train(args, env, kwargs)
    pass