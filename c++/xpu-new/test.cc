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
DEFINE_bool(print_outputs, false, "whether print output tensor elem, Default is false.");
DEFINE_bool(percision_test, false, "whether do percision diff with cpu config, Default is false.");
DEFINE_bool(profile_test, false, "whether do op profiling, Default is false.");

const int MAX_DISPLAY_OUTPUT_TENSOR_SIZE = 1000;

std::string shape_print(const std::vector<shape_t>& shapes) {
  std::string shapes_str{""};
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    auto shape = shapes[shape_idx];
    std::string shape_str;
    for (auto i : shape) {
      shape_str += std::to_string(i) + ",";
    }
    shapes_str += shape_str;
    shapes_str +=
        (shape_idx != 0 && shape_idx == shapes.size() - 1) ? "" : " : ";
  }
  return shapes_str;
}

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
   
  // performance methods
  // Enable profile tools to print op time
  if (FLAGS_profile_test)
    config.EnableProfile();

  // Debug methods
  /* [Debug]call train forward config
    1. if results is not same, yuan-sheng op has question. [TO TRIM MODEL]
    2. if results is same, inference config question
  */
  // paddle.static.Executor;

  // [Debug]close IR PASS optimization
  // config.SwitchIrOptim(false)

  // [Debug]Open IR debug to get intermedia pdmodel after every PASS
  // config.SwitchIrDebug();O

  // [Debug]Locate the specific Pass that caused the problem
  // config.pass_builder()->DeletePass("xxx_fuse_pass")
  return CreatePredictor(config);
}

void RunModel(Predictor *predictor, 
              std::vector<std::vector<float>> *out_datas) {
  auto output_names = predictor->GetOutputNames();

  // warm up
  double first_duration{-1};
  for (size_t i = 0; i < FLAGS_warmup; ++i) {
    if (i == 0) {
      auto warm_up_start = time();
      CHECK(predictor->Run());
      auto warm_up_end = time();
      first_duration = time_diff(warm_up_start, warm_up_end);
    } else {
      CHECK(predictor->Run());
    }
  }

  // millisecond;
  double sum_duration = 0.0;  
  double max_duration = 1e-5;
  double min_duration = 1e5;
  double avg_duration = -1;
  
  // repeats
  for (size_t i = 0; i < FLAGS_repeats; ++i) {
    auto st = time();
    CHECK(predictor->Run());
    auto duration = time_diff(st, time());
    if (i == 0 && first_duration < 0) {
      first_duration = duration;
    }
    sum_duration += duration;
    max_duration = duration > max_duration ? duration : max_duration;
    min_duration = duration < min_duration ? duration : min_duration;
    LOG(INFO) << "run_idx:  " << i + 1 << " / " << FLAGS_repeats 
              << ": " << duration << " ms";
  }
  avg_duration = sum_duration / static_cast<float>(FLAGS_repeats);
  LOG(INFO) << "\n======= Time Summary(ms) =======\n"
            << "1st_duration:" << first_duration << "\n"
            << "max_duration:" << max_duration << "\n"
            << "min_duration:" << min_duration << "\n"
            << "avg_duration:" << avg_duration << "\n";

  // output 
  std::vector<float> out_data;
  for (int j = 0; j < output_names.size(); j++) {
    out_data.clear();
    auto output_t = predictor->GetOutputHandle(output_names[j]);
    auto output_shape = output_t->shape();
    int output_t_numel = std::accumulate(output_shape.begin(), output_shape.end(), 
                                  1, std::multiplies<int>());
    output_t->CopyToCpu(out_data->data());
    LOG(INFO) << "\n====== output summary ======\n" 
              << "output shape(NCHW):" << shape_print(output_shape) << "\n"
              << "output tensor[" << j << "]'s elem num: " << output_t_numel << "\n"
              << " " << std::endl;
    // print output
    if (FLAGS_print_outputs) {
      for (int k = 0; k < output_t_numel && k < MAX_DISPLAY_OUTPUT_TENSOR_SIZE; ++k) {
        std::cout << "out_datas[" << j << "][" << k
                  << "]:" << out_data[k] << std::endl;
      }
    }
    out_datas->emplace_back(std::move(out_data));
  }
}

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto predictor = InitPredictor();

  if (FLAGS_model_dir.empty() &&
          (FLAGS_model_file.empty() || FLAGS_param_file.empty())) {
    std::cerr
        << "[ERROR] usage: \n"
        << argv[0] 
        << " --model_dir=                 string  Path to PaddlePaddle uncombined model.\n"
        << " --model_file=                string  Model file in PaddlePaddle combined model.\n"
        << " --param_file=                string  Param file in PaddlePaddle combined model.\n"
        << " --input_shapes=              string  input shapes of input tensor.\n"
        << " --warmup=1                   int32   Number of warmups.\n"
        << " --repeats=100                int32   Number of repeats.\n";
        << " --print_outputs=             bool    Print outputs tensor elem.\n";
        << " --percision_test=            bool    compare xpu-infer output with cpu-infer config.\n";
        << " --profile_test=              bool    Do op profiling of input model.\n";
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
  LOG(INFO) << "\n======= Model Messages =======\n"
            << "input_shape(s) (NCHW):" << shape_print(input_shapes) << "\n"
            << "model_dir:" << model_dir << "\n"
            << "warmup:" << warmup << "\n"
            << "repeats:" << repeats << "\n";
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

  std::vector<std::vector<float>> out_datas;
  // Run model
  RunModel(predictor.get(), &out_datas);

  if (FLAGS_percision_test) {
    float abs = 1e-15f;
    for (size_t i = 0; i < output_data_xpu.size(); i += 1) {
      for (size_t j = 0; j < output_data_xpu[i].size(); j += 1) {
        float diff = std::abs(output_data_cpu[i][j] - output_data_xpu[i][j]);
        if (diff > abs) abs = diff;
      }
    }
    LOG(INFO) << "Max abs = " << abs << std::endl;
  }

  return 0;
}
