from main import main_online_fintune, get_config_on

if __name__ == "__main__":
    args, env, kwargs = get_config_on()
    args.load_model = "/root/code/augDC/result/offline/08-07-16:09:26_PRDC_halfcheetah-medium-v2_1024/ckpt/PRDC_halfcheetah-medium-v2_1024_50000.pth"
    main_online_fintune(args, env, kwargs)
    pass