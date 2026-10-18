# Optional image classifier

I added this consumer to give the runtime work an actual inference application. It classifies a bounded set of digit images through a C++20 ONNX Runtime session and uses Adaptive Timing's existing admission, completion and generation reconciliation. The numerical runtime remains independently installable. It does not need ONNX Runtime, Pillow, a model download or these application dependencies.

`serial` remains the default and runs the original one-image graph once per image. Explicit `--execution-mode batched` uses a reviewed local graph derivation that preserves the batch dimension through flattening. A nonempty request of up to eight images makes one model invocation and returns every row as a completed, independently owned float32 `(N,10)` host array. Empty input returns `(0,10)` and makes no model invocation. The [current batching contract](../../docs/model-batching.md) records separate source and execution graph hashes, the unchanged trained weights and preprocessing, and the supported Python interfaces. This is one tensor for one logical request. It does not implement dynamic batching of independent requests or an asynchronous GPU API. Required CUDA continues to refuse CPU fallback.

The [accepted complete-request comparison](../../docs/batching-results.md) found lower two-image and eight-image request cost from batching within both providers. CPU still had the lower observed median at every tested batch size. The eight-image batched medians were 0.3635 ms on CPU and 0.6371 ms on required CUDA, while the one-image CPU comparison was inconclusive. The study includes in-memory image preparation through usable host-result reconciliation and excludes file decoding. The historical [serial contract and protocol](../../docs/inference-protocol.md), [CPU results](../../docs/inference-results.md) and [required CUDA follow-up](../../docs/inference-cuda-results.md) retain their original evidence and boundaries. The model card's Apache-2.0 metadata and MIT body/SPDX disagreement remains unresolved. Neither the original nor derived model bytes are included in the repository or wheel, and this local capability does not authorize model redistribution.

## Source and model

The [ONNX Model Zoo MNIST model](https://huggingface.co/onnxmodelzoo/mnist-8) is pinned to repository revision `a19f9a8c2333de1df9b03f10f5739f468b699a1a`. Its 26,454 bytes have SHA-256 `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`. The source describes CNTK training, ONNX IR 3 and opset 8. The card's metadata says Apache-2.0, while its SPDX comment and license section say MIT. The manifest retains that discrepancy. This repository does not redistribute the model or claim to resolve its artifact licensing. It also does not claim to reproduce the published training or accuracy result.

The [artifact manifest](artifacts.json) records the model and ONNX Runtime 1.30 API header identities. Header source is Microsoft's ONNX Runtime repository, under its [MIT license](https://github.com/microsoft/onnxruntime/blob/v1.30.0/LICENSE). Model Zoo's [Apache license](https://github.com/onnx/models/blob/main/LICENSE) applies separately. Models, headers and provider binaries are acquired into a chosen local directory, not committed here or included in the application wheel. Pillow, NumPy, ONNX and ONNX Runtime retain their own licenses. The application source uses the repository's MIT license.

The generated example digits are original geometric drawings. They demonstrate decoding and completed predictions. They are deliberately labelled examples, not a held-out dataset. Incorrect geometric-digit predictions are retained rather than filtered.

## Build separately

Use Python 3.11, CMake 3.24 or later and a C++20 compiler. On Windows, run from a developer terminal with MSVC available. The following commands are issued from the repository root. Choose an artifact directory outside the working tree if it may contain private files. The example below uses an ignored application directory.

```sh
python -m venv .venv-inference
# Activate the environment using your platform's normal command.
python -m pip install scikit-build-core==0.11.6 pybind11==3.1.0 cmake==3.31.6 ninja==1.13.0
python examples/image_classifier/acquire.py examples/image_classifier/.artifacts
python -m pip install "./examples/image_classifier[cpu,test]" -Ccmake.define.HBR_ORT_INCLUDE="$PWD/examples/image_classifier/.artifacts/sdk/include"
```

The optional package declares its Adaptive Timing dependency at the exact public revision shown in the verification report. Installation therefore also needs ordinary public Git access. PowerShell uses the same final setting with `$PWD` converted to its full path in the quoted argument. Keep build parallelism at one on a shared machine. The exact tested versions and actual executions appear in the [verification report](../../docs/inference-results.md).

For a separate installed application, set `HBR_MNIST_MODEL` to the acquired model file and run the test directory. This variable is a test input, not a hidden model download.

```sh
HBR_MNIST_MODEL="$PWD/examples/image_classifier/.artifacts/mnist-8.onnx" python -m pytest examples/image_classifier/tests -q
hbr-classify --model examples/image_classifier/.artifacts/mnist-8.onnx --provider cpu --profile-prefix ort-profile --output predictions.json
```

Pass up to eight PNG paths after the options to classify supplied images. PNG files must contain one frame, be at most one MiB and have dimensions at most 256 by 256. Images are composited over black, converted to grayscale, resized to 28 by 28 with bilinear sampling and divided by 255. Use black-background, white-foreground digits. The model is not an arbitrary photograph classifier.

Use explicit batched execution to send the admitted images through one model invocation:

```sh
hbr-classify --model examples/image_classifier/.artifacts/mnist-8.onnx --provider cpu --execution-mode batched --profile-prefix ort-profile-batched --output predictions-batched.json
```

The later [completed-request cost investigation](../../docs/request-cost-analysis.md) preserves its qualified no-optimization decision. Its host `Session.run` interval includes several costs, and ordinary operation traces did not resolve device activity. Neither that investigation nor a lower isolated interval establishes a new completed-request speedup.

## Provider and lifecycle limits

The CPU installation uses ONNX Runtime's CPU package and API version 30. Use a separate environment for ONNX Runtime GPU 1.30.0 and its compatible CUDA and cuDNN dependencies. The official package supports `onnxruntime-gpu[cuda,cudnn]==1.30.0`. Its vendor libraries remain outside the default installation.

The automatic `cuda-required` path calls ONNX Runtime's documented `preload_dlls(directory="")` operation to resolve the installed NVIDIA wheels before creating the C++ session. It leaves system paths and settings unchanged. An explicitly supplied runtime library retains the caller's dependency loading contract. Required CUDA disables CPU fallback and fails visibly if CUDA cannot initialize. Available provider names alone do not prove operation placement. The [CUDA protocol](../../docs/inference-cuda-protocol.md) defines the separate actual output, placement, memory and comparison evidence.

Each request owns its snapshot and results. An immutable generation does not cancel physical work. Running cancellation prevents adoption while synchronous inference finishes. Queued cancellation can stop admission. Closing waits for work and closes the session on its owner worker. Capacity includes completed results until reconciliation. Close the executor after at most 256 admitted requests and start a new session for a longer run. Errors do not reuse a request identity or silently adopt a stale result.

The consumer does not install GPU dependencies, change counter permissions, reset a device or select an unverified vendor backend. Keep provider acquisition and native GPU acceptance separate from the default CPU installation.

The repository's `tools/check_inference_evidence.py` and `tools/check_inference_cuda_evidence.py` check retained execution identities. Later source changes are compared with the recorded historical Git snapshot, so those checks require Git and the recorded commit objects. Use a full repository clone, or fetch the recorded source revision into a shallow clone. A downloaded source archive can run the current application and its contracts, but lacks the history needed to verify changed historical source. Hosted consumer checks fetch that history explicitly. On a compatible, separately configured GPU environment, set `HBR_INFERENCE_GPU_ACCEPTANCE=1` to explicitly run the serial and batched required-CUDA cases in `test_gpu_inference.py`. Ordinary CPU checks skip them.

When a performance slot is free, `compare.py --model <reviewed-model> --output <new-results.json>` executes the separately committed prepared-input protocol. It retains twenty complete-call samples per condition and generates its table from those rows. It is not run during an occupied shared-workstation slot. Its host intervals do not establish device time or total process/device memory. No comparison table is presented until real collection completes.
