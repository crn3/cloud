from pathlib import Path
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    onnx_path = str(Path(onnx_path).resolve())
    engine_path = str(Path(engine_path).resolve())

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")
    else:
        print("FP16 not enabled")

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        success = parser.parse(f.read())

    if not success:
        print("ONNX parse failed. Errors:")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return False

    print("Parsed ONNX successfully")
    print(f"Network inputs:  {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"Input {i}: name={tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}")

    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"Output {i}: name={tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Engine build failed")
        return False

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Saved engine: {engine_path}")
    return True

if __name__ == "__main__":
    build_engine("Unet.onnx", "Unet_fp16.engine", fp16=True)
    build_engine("UnetPlusPlus.onnx", "UnetPlusPlus_fp16.engine", fp16=True)
