def get_checkpoint_dir(version):
    # 根据version值选择checkpoint路径
    if version == 0:
        dir = "checkpoints/resnet18_pretrain_test-epoch=44-val_loss=0.24.ckpt"   #res18最佳checkpoint保存路径
    elif version == 1:
        dir = "checkpoints/resnet34_pretrain_test-epoch=14-val_loss=0.05.ckpt"   #res34最佳checkpoint保存路径
    else:
        raise ValueError("Unsupported version: {}".format(version))
    
    return dir

def get_log_name(version):
    # 根据version值设置模型名称
    if version == 0:
        model_name = "resnet18"
    elif version == 1:
        model_name = "resnet34"
    else:
        raise ValueError("Unsupported version: {}".format(version))
    
    # 使用字符串格式化方法替换模型名称
    log_name = "{}_pretrain_test".format(model_name)
    return log_name
