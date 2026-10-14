// This file is responsible for checking if the logits from each layer produced from generate.cpp
// are correct. To do this, I first downloaded the reference gpt2 implementation from Karpathy's
// llm.c repo. Then generate gpt2_124M_debug_state.bin by running train_gpt2.py with --write_tensors
// 1 option. gpt2_124M_debug_state.bin is nothing but the logits dumped for one forward pass of the
// model. The gpt2_124M_debug_state.bin has the following format:
//
// ** FILE FORMAT (little endian) **
// header: 0-1024
//   - magic 20240327, version 2, B=4, T=64
// x: 1024-2048 (4 sequences/batch * 64 tokens / sequence * 4 bytes/token = 1204 bytes/batch)
// other fields such as expected the predicated label for each x (y), the final logits, the loss and
// the gradients.
// ** END FILE FORMAT **
//
// Crucially, it does not dump out the intermediate activations for each of the 12 layers.
// To get these intermediate activations, I instrumented test_gpt2.c to dump these out.
// test_gpt2.c also checks if the final logits of the C implementation match with the python
// reference GPT2 implementation. The intermediate activations are available for download in this
// repo (reference_activations.bin), so they need not be re-generated. I also extracted 64 x
// values (these are tokenids from the tinyshakespeare dataset that Karpathy uses) and included it
// in this repo (tokens.bin) so that this entire repo is self-contained.
//
// ** tokens.bin FILE FORMAT (little endian) **
//
// Input token ids extracted from llm.c's gpt2_124M_debug_state.bin, so the
// repo is self-contained for activation comparison.
//
// header: bytes 0-255
//   [0..3]   magic          value = 20260927 (uint32)
//   [4]      version        uint8, = 1
//   [5..7]   padding        zero -- for aligned loads
//   [8..11]  token count    uint32, = 64
//   [12..255] padding       zero
//
// payload: bytes 256 onwards
//   token count x int32 token ids
//
// Total size = 256 + (token count * 4) = 512 bytes.
//
// The 64 ids are llm.c's x tensor, shaped (B=4 (rows), T=64 (cols)) row-major.
// Here we use only the first 64 tokens of the 1st minibatch (row = 0).
//
//
// ** reference_activations.bin FILE FORMAT (little endian, fp32) **
//
// Reference activations from Karpathy's llm.c (test_gpt2.c), step-0 forward pass
// over the 64 token ids in tokens.bin. Batch 0 only. llm.c's own logits, loss and
// gradient checks against the PyTorch reference passed on the producing run.
//
// header: bytes 0-1023 (256 x int32)
//   [0] magic       = 20260928
//   [1] version     = 1
//   [2] B           = 1
//   [3] T           = 64
//   [4] C           = 768
//   [5] L           = 12
//   [6] Vp          = 50304   (PADDED vocab -- logits are this wide, not 50257)
//   [7] num_tensors = 15
//   [8] logit_rows  = 1       (tensor 14 holds ONLY position T-1)
//   [9..255] zero
//
// payload: bytes 1024 onwards. 15 tensors back to back, no per-tensor headers.
// Tensors 0..13 are row-major fp32, shape (T, C): element (t,c) at t*C + c.
// Tensor 14 is a single row of Vp floats -- the logits at position T-1 only.
//
//   idx  tensor                             shape       floats    byte offset
//   ---  ---------------------------------- ----------  --------  -----------
//     0  after token+position embeddings    (64, 768)     49,152        1,024
//     1  after layer 0                      (64, 768)     49,152      197,632
//     2  after layer 1                      (64, 768)     49,152      394,240
//     3  after layer 2                      (64, 768)     49,152      590,848
//     4  after layer 3                      (64, 768)     49,152      787,456
//     5  after layer 4                      (64, 768)     49,152      984,064
//     6  after layer 5                      (64, 768)     49,152    1,180,672
//     7  after layer 6                      (64, 768)     49,152    1,377,280
//     8  after layer 7                      (64, 768)     49,152    1,573,888
//     9  after layer 8                      (64, 768)     49,152    1,770,496
//    10  after layer 9                      (64, 768)     49,152    1,967,104
//    11  after layer 10                     (64, 768)     49,152    2,163,712
//    12  after layer 11                     (64, 768)     49,152    2,360,320
//    13  after the final layernorm          (64, 768)     49,152    2,556,928
//    14  logits at position 63 only         (1, 50304)    50,304    2,753,536
//
// Tensor 0 is llm.c's "encoded"; 1..12 are "residual3[l]", 13 is "lnf"; 14 is "logits".
// Take a look at the model.acts structure in llm.c/train_gpt2.c - this is where the
// activations above come from.
// ** END FILE FORMAT **
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <variant>

#include "checkpoint.hpp"
#include "decoder.hpp"
#include "mappedfile.hpp"
#include "reference.hpp"

using decoder::Decoder;
using namespace std;

static chrono::time_point<chrono::steady_clock> start;

// see https://docs.pytorch.org/docs/2.14/generated/torch.allclose.html
bool allClose(vector<float> a, span<const float> b, double rtol = 1e-05, double atol = 1e-3) {
    if (a.size() != b.size()) return false;
    return equal(a.begin(), a.end(), b.begin(), [rtol, atol](float val_a, float val_b) {
        if (isnan(val_a) || isnan(val_b)) {
            return false;
        }
        return abs(val_a - val_b) <= (atol + rtol * abs(val_b));
    });
}

void checkActivations(const int index, const Tensor& t) {
    auto end = chrono::steady_clock::now();
    static constexpr auto REFERENCEFILE = "../reference_activations.bin";
    static Reference ref(REFERENCEFILE);
    if (index == 0)
        cout << "Checking encodings ...";
    else if (index >= 1 && index <= 12)
        cout << "Checking activations for layer " << setw(2) << index << " ...";
    else if (index == 13)
        cout << "Checking activations for final layernorm ...";
    else if (index == 14)
        cout << "Checking activations for final vocab projection ...";

    if (!allClose(t.flatten(), ref.spanAtIndex(index))) {
        // Dump a few activations
        cout << "\nGot:";
        for (int i = 0; i < 10; i++) {
            cout << fixed << setprecision(7) << setw(10) << t.flatten()[i] << " ";
        }
        cout << "\nRef:";
        for (int i = 0; i < 10; i++) {
            cout << fixed << setprecision(7) << setw(10) << ref.spanAtIndex(index)[i] << " ";
        }
        cout << "\n";
        cerr << "Failed to compare with reference. Layer failed at: " << index << "\n";
        assert(0);
    } else {
        cout << " PASS ";
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Time taken: " << duration << "ms\n";
        start = end;
    }
}

void printHelp(char* programName) {
    cout << "Usage: " << programName;
    cout << " [-h | --help] <weights_file> <sample_tokens_file>" << "\n";
    cout << "Notes: See karpathy/llm.c to get the GPT2_124M.bin weights file.\n";
    cout << "       This program cannot read the HuggingFace safetensors format.\n";
}

void parseArgs(unordered_map<string, variant<int, string>>& args, int argc, char** argv) {
    vector<string> positional;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            exit(0);
        }
        if (!arg.empty() && arg[0] == '-') {
            cerr << "error: unknown option '" << arg << "'\n";
            printHelp(argv[0]);
            exit(1);
        }
        positional.push_back(arg);
    }

    // Mandatory positional args
    if (positional.size() != 2) {
        printHelp(argv[0]);
        exit(positional.empty() ? 0 : 1);
    }
    args["weights_file"] = positional[0];
    args["sample_tokens_file"] = positional[1];
}

// Usage: testactivations [-h | --help] <weights_file> <sample_tokens_file>
int main(int argc, char** argv) {
    // constexpr auto CHECKPOINTFILE = "../gpt2_124M.bin";
    // constexpr auto TOKENIDFILE = "../tokens.bin";
    // construct the checkpoint
    unordered_map<string, variant<int, string>> args;
    parseArgs(args, argc, argv);
    Checkpoint ckpt(get<string>(args["weights_file"]).c_str());
    cout << "Loaded checkpoint file. Size: " << ckpt.mp.size() / (1024 * 1024) << "MB\n";
    cout << "Context length:         " << ckpt.maxT << "\n";
    cout << "Vocabulary size:        " << ckpt.vocab << "\n";
    cout << "Layers:                 " << ckpt.layers << "\n";
    cout << "Attention heads:        " << ckpt.nh << "\n";
    cout << "Embedding dimensions:   " << ckpt.embsz << "\n";
    cout << "Padded vocabulary size: " << ckpt.vocab_padded << "\n";
    cout << "----------------------------------------\n";
    // instantiate the decoder from the checkpoint
    Decoder gpt2(ckpt, false /* use_kv_cache */);

    // read tokens from file to pass into the decoder
    MappedFile tf(get<string>(args["sample_tokens_file"]).c_str());
    constexpr auto OFFSET_MAGIC = 0;
    constexpr auto OFFSET_VERSION = 4;
    constexpr auto OFFSET_TOKENCOUNT = 8;
    constexpr auto OFFSET_TOKENIDS = 256;

    constexpr auto MAGIC = 20260927;
    constexpr auto VERSION = 1;

    assert(tf.to_int32(tf.bytes().subspan(OFFSET_MAGIC, sizeof(int32_t))) == MAGIC);
    assert(tf.to_int32(tf.bytes().subspan(OFFSET_VERSION, sizeof(int8_t))) == VERSION);

    int tokencount = tf.to_int32(tf.bytes().subspan(OFFSET_TOKENCOUNT, sizeof(int32_t)));

    // now open the reference file
    span<const int> tokenids = tf.span_at<const int>(OFFSET_TOKENIDS, tokencount);
    // since forward takes in a vector, we need to convert the span to a vector
    vector<int> v;
    v.assign(tokenids.begin(), tokenids.end());
    start = chrono::steady_clock::now();
    Tensor y = gpt2.forward(v, checkActivations);
}