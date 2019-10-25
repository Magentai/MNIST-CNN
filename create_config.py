import configparser as cp
 
def createConfig(path):
    config = cp.ConfigParser()
    config.add_section("pathes")
    config.set("pathes", "path_to_model", "model/model.h5")
    config.set("pathes", "path_to_test_data", "data/")
    config.set("pathes", "path_to_json", "result.json")
    config.add_section("constants_for_cnn")
    config.set("constants_for_cnn", "batch_size", "128")
    config.set("constants_for_cnn", "num_classes", "10")
    config.set("constants_for_cnn", "epochs", "2")
    config.set("constants_for_cnn", "part_val", "0.16")
    config.set("constants_for_cnn", "img_rows", "28")
    config.set("constants_for_cnn", "img_cols", "28")
    
    with open(path, "w") as config_file:
        config.write(config_file)
 
createConfig("config.ini")