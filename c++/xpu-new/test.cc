#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;
using paddle_infer::PrecisionType;
using shape_t = std::vector<int>;
using Time = decltype(std::chrono::high_resolution_clock::now());

DEFINE_string(model_file, "", "Directory of the inference model.");
DEFINE_string(param_file, "", "Directory of the inference model.");
DEFINE_string(model_dir, "", "Directory of the inference model.");
DEFINE_string(input_shapes, "1,3,224,224", "shapes of model inputs.");
DEFINE_int32(warmup, 5, "warmup.");
DEFINE_int32(repeats, 20, "repeats.");

const int MAX_DISPLAY_OUTPUT_TENSOR_SIZE = 10000;

std::vector<std::string> split_string(const std::string& str_in) {
  std::vector<std::string> str_out;
  std::string tmp_str = str_in;
  while (!tmp_str.empty()) {
    size_t next_offset = tmp_str.find(":");
    str_out.push_back(tmp_str.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return str_out;
}

std::vector<int> get_shape(const std::string& str_shape) {
  std::vector<int> shape;
  std::string tmp_str = str_shape;
  while (!tmp_str.empty()) {
    int dim = atoi(tmp_str.data());
    shape.push_back(dim);
    size_t next_offset = tmp_str.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return shape;
}

Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

std::shared_ptr<Predictor> InitPredictor() {
  Config config;
  if (FLAGS_model_dir != "") {
    config.SetModel(FLAGS_model_dir);
  }
  config.SetModel(FLAGS_model_file, FLAGS_param_file);

  // Enable Kunlun XPU
  config.EnableXpu();

  // Open the memory optim.
  config.EnableMemoryOptim();

  // 开启 IR 打印
  config.SwitchIrDebug();
  return CreatePredictor(config);
}

void run(Predictor *predictor, 
         std::vector<float> *out_data) {
  auto output_names = predictor->GetOutputNames();
  // warm up
  for (size_t i = 0; i < FLAGS_warmup; ++i)
    CHECK(predictor->Run());
  // repeats
  auto st = time();
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    CHECK(predictor->Run());
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    auto output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                  std::multiplies<int>());
    out_data->resize(out_num);
    output_t->CopyToCpu(out_data->data());
  }
  LOG(INFO) << "run avg time is " << time_diff(st, time()) / FLAGS_repeats
            << " ms";
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();

  if (FLAGS_model_dir.empty() &&
          (FLAGS_model_file.empty() || FLAGS_param_file.empty())) { //读取所有参数
    std::cerr
        << "[ERROR] usage: \n"
        << argv[0] << " --model_dir=                 string  Path to "
                      "PaddlePaddle uncombined model.\n"
        << " --model_file=                string  Model file in PaddlePaddle "
           "combined model.\n"
        << " --param_file=                string  Param file in PaddlePaddle "
           "combined model.\n"
        << " --input_shapes=               string  input shapes of "
           "input tensor.\n"
        << " --warmup=10                  int32   Number of warmups.\n"
        << " --repeats=100                int32   Number of repeats.\n";
    exit(1);
  }

  // Prepare input data
  std::vector<std::string> str_input_shapes;
  std::vector<shape_t> input_shapes{
      {1, 3, 224, 224}};  // shape_t ==> std::vector<int64_t>
  std::string raw_input_shapes = FLAGS_input_shapes;
  std::cout << "raw_input_shapes: " << raw_input_shapes << std::endl;
  if (!FLAGS_input_shapes.empty()) {
    str_input_shapes = split_string(raw_input_shapes);
    input_shapes.clear();
    for (size_t i = 0; i < str_input_shapes.size(); ++i) {
      std::cout << "input shape: " << str_input_shapes[i] << std::endl;
      input_shapes.push_back(get_shape(str_input_shapes[i]));
    }
  }
  // prepare input data
  std::cout << "input_shapes.size():" << input_shapes.size() << std::endl;
  auto input_names = predictor->GetInputNames();
  int input_num = 1;
  for (int i = 0; i < input_shapes.size(); ++i) {
    input_num = 1;
    auto input_tensor = predictor->GetInputHandle(input_names[i]);
    input_tensor->Reshape(input_shapes[i]);
    // auto input_data = input_tensor->mutable_data<float>();
    for (int j = 0; j < input_shapes[i].size(); ++j) {
      input_num *= input_shapes[i][j];
    }
    std::vector<float> input_data(input_num);
    // all 1.f
    for (int k = 0; k < input_num; ++k) {
      input_data[k] = 1.f;
    }
    input_tensor->CopyFromCpu(input_data.data());
  }

  std::vector<float> out_data;
  // Run model
  run(predictor.get(), &out_data);

  for (size_t i = 0; i < out_data.size() && i < MAX_DISPLAY_OUTPUT_TENSOR_SIZE; i ++) {
    LOG(INFO) << "[" << i << "] " << out_data[i] << std::endl;
  }
  return 0;
}
