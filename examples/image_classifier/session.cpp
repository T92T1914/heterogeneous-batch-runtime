#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace py = pybind11;
namespace {
// One explicitly supplied runtime library is retained for the process. This
// keeps the API table alive until all sessions and Python objects are gone.
void initialize(const std::string& library) {
  static std::mutex mutex;
  static std::string loaded;
  std::lock_guard lock(mutex);
  const auto path = std::filesystem::weakly_canonical(library);
  if (!path.is_absolute() || !std::filesystem::is_regular_file(path))
    throw std::invalid_argument("runtime library must be an existing absolute file");
  if (!loaded.empty()) {
    if (loaded != path.string()) throw std::invalid_argument("one runtime library per process");
    return;
  }
#ifdef _WIN32
  const auto handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (!handle) throw std::runtime_error("cannot load supplied ONNX Runtime library");
  const auto entry = reinterpret_cast<const OrtApiBase* (ORT_API_CALL*)()>(GetProcAddress(handle, "OrtGetApiBase"));
#else
  const auto handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw std::runtime_error("cannot load supplied ONNX Runtime library");
  const auto entry = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(handle, "OrtGetApiBase"));
#endif
  const auto api = entry ? entry()->GetApi(ORT_API_VERSION) : nullptr;
  if (!api) {
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
    throw std::runtime_error("runtime does not provide the required API version 30");
  }
  Ort::InitApi(api);
  loaded = path.string();
}

struct State {
  Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "hbr-image-classifier"};
  Ort::SessionOptions options;
  Ort::Session session{nullptr};
  std::string input, output;
  std::vector<char> model;
  State(std::string bytes, const std::string& provider, const std::string& profile, bool batched) : model(bytes.begin(), bytes.end()) {
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == "cuda-required") {
      // A requested CUDA provider is not permission to silently run on CPU.
      OrtCUDAProviderOptionsV2* raw = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw));
      const auto release = [](OrtCUDAProviderOptionsV2* x) { Ort::GetApi().ReleaseCUDAProviderOptions(x); };
      std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(release)> cuda(raw, release);
      const char* keys[]{"device_id", "gpu_mem_limit", "arena_extend_strategy", "do_copy_in_default_stream", "use_tf32", "cudnn_conv_use_max_workspace"};
      const char* values[]{"0", "134217728", "kSameAsRequested", "1", "0", "0"};
      Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(raw, keys, values, 6));
      options.AppendExecutionProvider_CUDA_V2(*raw);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    } else if (provider != "cpu") {
      throw std::invalid_argument("provider must be cpu or cuda-required");
    }
    if (!profile.empty()) options.EnableProfiling(std::filesystem::path(profile).c_str());
    session = Ort::Session(environment, model.data(), model.size(), options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
      throw std::invalid_argument("model must have one input and one output");
    auto input_type = session.GetInputTypeInfo(0);
    auto output_type = session.GetOutputTypeInfo(0);
    auto in = input_type.GetTensorTypeAndShapeInfo();
    auto out = output_type.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> expected_input{batched ? -1 : 1,1,28,28};
    const std::vector<int64_t> expected_output{batched ? -1 : 1,10};
    if (in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || in.GetShape() != expected_input
        || out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || out.GetShape() != expected_output)
      throw std::invalid_argument("selected float32 MNIST graph input and logits contract required");
    Ort::AllocatorWithDefaultOptions allocator;
    input = session.GetInputNameAllocated(0, allocator).get();
    output = session.GetOutputNameAllocated(0, allocator).get();
  }
};

class Session {
  std::thread::id owner_ = std::this_thread::get_id();
  std::unique_ptr<State> state_;
  bool profiling_;
  const bool batched_;
  std::uint64_t model_invocations_ = 0;
  void check() const {
    if (owner_ != std::this_thread::get_id()) throw std::runtime_error("session operations require their owning thread");
    if (!state_) throw std::runtime_error("session is closed");
  }
 public:
  Session(const std::string& library, py::bytes bytes, const std::string& provider, const std::string& profile, bool batched)
      : profiling_(!profile.empty()), batched_(batched) {
    const std::string model = bytes;
    if (model.empty() || model.size() > 1048576) throw std::invalid_argument("model byte limit exceeded");
    py::gil_scoped_release release;
    initialize(library);
    state_ = std::make_unique<State>(model, provider, profile, batched_);
  }
  py::array_t<float> run(const py::array_t<float, py::array::c_style>& batch) {
    check();
    if (batch.ndim() != 4 || batch.shape(1) != 1 || batch.shape(2) != 28 || batch.shape(3) != 28 || batch.shape(0) > 8)
      throw std::invalid_argument("batch must have shape (N,1,28,28), N <= 8");
    // Explicit copy before releasing the GIL. External native writers must be
    // excluded during this copy. Successful return never aliases caller input.
    // C contiguous input can still be unaligned, so access its bytes directly.
    std::vector<float> owned(static_cast<std::size_t>(batch.size()));
    if (!owned.empty()) {
      const py::array& input = batch;
      std::memcpy(owned.data(), input.data(), owned.size() * sizeof(float));
    }
    for (float x : owned) if (!std::isfinite(x) || x < 0 || x > 1)
      throw std::invalid_argument("finite normalized pixels in [0,1] required");
    py::array_t<float> result({batch.shape(0), py::ssize_t{10}});
    auto* destination = result.mutable_data();
    const auto count = batch.shape(0);
    {
      py::gil_scoped_release release;
      auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      const char* inputs[]{state_->input.c_str()};
      const char* outputs[]{state_->output.c_str()};
      const auto calls = batched_ && count ? py::ssize_t{1} : count;
      for (py::ssize_t i = 0; i < calls; ++i) {
        const auto tensor_batch = batched_ ? count : py::ssize_t{1};
        const std::array<int64_t, 4> shape{tensor_batch,1,28,28};
        const auto input_offset = batched_ ? py::ssize_t{0} : i*784;
        auto tensor = Ort::Value::CreateTensor<float>(memory, owned.data() + input_offset,
            static_cast<std::size_t>(tensor_batch*784), shape.data(), shape.size());
        ++model_invocations_;
        auto values = state_->session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
        if (values.size() != 1 || !values[0].IsTensor()) throw std::runtime_error("unexpected output tensor");
        auto info = values[0].GetTensorTypeAndShapeInfo();
        const auto elements = static_cast<std::size_t>(tensor_batch*10);
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != std::vector<int64_t>{tensor_batch,10} || info.GetElementCount() != elements)
          throw std::runtime_error("unexpected output shape or dtype");
        std::copy_n(values[0].GetTensorData<float>(), elements, destination + i*10);
      }
      // Ordinary Run returns completed outputs. No I/O binding, caller stream
      // or disable_synchronize_execution_providers option is used here.
      for (py::ssize_t i = 0; i < count*10; ++i)
        if (!std::isfinite(destination[i])) throw std::runtime_error("nonfinite model output");
    }
    return result;
  }
  // Counts attempted actual Run calls, after input admission. Empty input adds none.
  std::uint64_t model_invocations() const { check(); return model_invocations_; }
  std::string close() {
    if (owner_ != std::this_thread::get_id()) throw std::runtime_error("close requires the owning thread");
    if (!state_) return {};
    std::string profile;
    {
      py::gil_scoped_release release;
      if (profiling_) {
        Ort::AllocatorWithDefaultOptions allocator;
        profile = state_->session.EndProfilingAllocated(allocator).get();
      }
      state_.reset();
    }
    return profile;
  }
};
}
PYBIND11_MODULE(_session, module) {
  py::class_<Session>(module, "Session")
      .def(py::init<const std::string&,py::bytes,const std::string&,const std::string&,bool>(),
           py::arg("library"), py::arg("model"), py::arg("provider"), py::arg("profile"), py::arg("batched") = false)
      .def("run", &Session::run, py::arg("batch").noconvert())
      .def_property_readonly("model_invocations", &Session::model_invocations)
      .def("close", &Session::close);
}
