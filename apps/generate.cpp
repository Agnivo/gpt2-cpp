#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "checkpoint.hpp"
#include "decoder.hpp"
#include "mappedfile.hpp"
#include "tensor.hpp"
#include "utils.hpp"

using namespace std;
using decoder::Decoder;
using tn::Tensor;

// Format of the file containing the id -> token mappings
// Little endian
// 256 x int32 little-endian header:
//     [0] magic 20240328
//     [1] version 2 (v2 adds the EOT token id at [3])
//     [2] vocab size n
//     [3] EOT token id
// then, for each token id 0..n-1:
//     1 byte length, followed by that many raw bytes
constexpr auto OFFSET_MAGIC = 0;                       // Magic number
constexpr auto OFFSET_VERSION = 1 * sizeof(int32_t);   // Version
constexpr auto OFFSET_VOCAB = 2 * sizeof(int32_t);     // Vocab size
constexpr auto OFFSET_EOT = 3 * sizeof(int32_t);       // End of token ID
constexpr auto OFFSET_TOKENS = 256 * sizeof(int32_t);  // Start of [len][...tokenbytes...]

constexpr auto MAGIC = 20240328;
constexpr auto VERSION = 2;

string idToToken(int id, MappedFile& tk, size_t startofTokens) {
    size_t offset = startofTokens;
    for (int i = 0; i < id; i++) {
        // Read length byte
        uint8_t len = static_cast<uint8_t>(tk.bytes()[offset++]);
        offset += len;
    }
    return string(reinterpret_cast<const char*>(tk.bytes().data() + offset + 1),
                  static_cast<uint8_t>(tk.bytes()[offset]));
}

void printHelp(char* programName) {
    cout << "Usage: " << programName;
    cout << " [-h | --help] [-t | --temperature] <weights_file> <tokenizer_file> <num_tokens> "
            "<prompt>\n";
    cout << "Notes: See karpathy/llm.c to get the GPT2_124M.bin weights file.\n";
    cout << "       This program cannot read the HuggingFace safetensors format.\n";
}

void parseArgs(unordered_map<string, variant<bool, int, string, float>>& args, int argc,
               char** argv) {
    // meaningful defaults
    args["temperature"] = 0.75f;
    args["use_kv_cache"] = false;

    vector<string> positional;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            exit(0);
        }
        if (arg == "-t" || arg == "--temperature") {
            if (i + 1 >= argc) {
                cerr << "error: " << arg << " requires a value\n";
                printHelp(argv[0]);
                exit(1);
            }
            float temperature = 0.75f;
            if (!convert_arg(string(argv[++i]), temperature)) {
                cerr << "error: could not parse temperature '" << argv[i] << "'\n";
                exit(1);
            }
            if (temperature < 0.0f) temperature = 0.75f;
            args["temperature"] = temperature;
            continue;
        }
        if (arg == "-kv") {
            args["use_kv_cache"] = true;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            cerr << "error: unknown option '" << arg << "'\n";
            printHelp(argv[0]);
            exit(1);
        }
        positional.push_back(arg);
    }

    // Mandatory positional args
    if (positional.size() != 4) {
        printHelp(argv[0]);
        exit(positional.empty() ? 0 : 1);
    }
    args["weights_file"] = positional[0];
    args["tokenizer_file"] = positional[1];
    int num_tokens = 0;
    if (!convert_arg(positional[2], num_tokens) || num_tokens <= 0) {
        cerr << "error: num_tokens must be a positive integer\n";
        exit(1);
    }
    args["num_tokens"] = num_tokens;
    args["prompt"] = positional[3];
}

int main(int argc, char** argv) {
    // parse command line
    unordered_map<string, variant<bool, int, string, float>> args;
    parseArgs(args, argc, argv);
    cout << "Weights file:          " << get<string>(args["weights_file"]) << "\n";
    cout << "Tokenizer file:        " << get<string>(args["tokenizer_file"]) << "\n";
    cout << "Tokens to generate:    " << get<int>(args["num_tokens"]) << "\n";
    cout << "Temperature            " << get<float>(args["temperature"]) << "\n\n";

    MappedFile tk = MappedFile(get<string>(args["tokenizer_file"]).c_str());  // open the token file
    assert(tk.to_int32(tk.bytes().subspan(OFFSET_MAGIC, sizeof(int32_t))) == MAGIC);
    assert(tk.to_int32(tk.bytes().subspan(OFFSET_VERSION, sizeof(int32_t))) == VERSION);
    int32_t vocab = tk.to_int32(tk.bytes().subspan(OFFSET_VOCAB, sizeof(int32_t)));
    int32_t eot = tk.to_int32(tk.bytes().subspan(OFFSET_EOT, sizeof(int32_t)));

    // "Once upon a time, there was a"
    vector<int32_t> tokens = {7454, 2402, 257, 640, 11, 612, 373, 257};
    int token_pos;
    vector<int32_t> output;

    Checkpoint ckpt(get<string>(args["weights_file"]).c_str());
    cout << "Loaded checkpoint file. Size: " << ckpt.mp.size() / (1024 * 1024) << "MB\n";
    cout << "Context length:         " << ckpt.maxT << "\n";
    cout << "Vocabulary size:        " << ckpt.vocab << "\n";
    cout << "Layers:                 " << ckpt.layers << "\n";
    cout << "Attention heads:        " << ckpt.nh << "\n";
    cout << "Embedding dimensions:   " << ckpt.embsz << "\n";
    cout << "Padded vocabulary size: " << ckpt.vocab_padded << "\n";

    // Autoregressive loop
    int32_t pred = tokens[0];
    int tokens_generated = 0;
    int numtokens = get<int>(args["num_tokens"]);
    float temp = get<float>(args["temperature"]);
    bool use_kv_cache = get<bool>(args["use_kv_cache"]);
    constexpr int TOPK = 40;
    constexpr unsigned int SEED = 42;
    std::mt19937 gen(SEED);  // random number generator
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // instantiate the decoder from the checkpoint
    Decoder gpt2(ckpt, use_kv_cache);
    Tensor logits;
    auto start_time = std::chrono::steady_clock::now();

    if (use_kv_cache) {
        vector<int32_t> prefill_tokens = vector<int32_t>(tokens.begin(), tokens.end() - 1);
        gpt2.prefill(prefill_tokens);
    }

    token_pos = tokens.size() - 1;
    while (true) {
        if (use_kv_cache) {
            logits = gpt2.decode(tokens[tokens.size() - 1], token_pos, nullptr);  // shape (1, V)
        } else {
            logits = gpt2.forward(tokens, nullptr);  // shape (1, V)
        }

        if (temp == 0.0f) {
            // greedy decoding
            pred = logits.argmax(0);
        } else {
            pred = -1;
            // divide all logits by temperature
            logits = logits / get<float>(args["temperature"]);
            // sampling algo:
            // -----
            // 1. sort the logits vector conceptually (i.e. store permuted indexes into the logits
            // vector in vector idx - this permutation is in the decreasing order of the logits)
            // example: if logits = [1.00, 3.40, 4.00] then idx = [2, 1, 0]
            // 2. pick top k logits
            // 3. softmax to get probabilities
            // 4. pick a random number from uniform distribution [0, 1)
            // 5. sum the probabilities starting from the beginning until it is >= the random numer
            // picked
            // 6. the corresponding idx is the pred value
            // ----
            // step 1
            vector<size_t> idx;
            idx = logits.argsort();
            // step 2
            Tensor topk(1, TOPK);
            for (int i = 0; i < TOPK; i++) topk.set(0, i, logits.at(0, idx[i]));
            // step 3
            topk = topk.softmax();
            // step 4
            float random_value = dis(gen);
            // step 5 and 6
            float sum = 0.0f;
            for (int i = 0; i < topk.cols(); i++) {
                sum += topk.at(0, i);
                if (sum >= random_value) {
                    pred = idx[i];
                    break;
                }
            }
            // corner case guard: if we fell off the loop without sum >= random_value), then pick
            // the last bucket
            if (pred == -1) pred = idx[TOPK - 1];
        }

        if ((pred != eot) && (tokens_generated++ < numtokens)) {
            assert(pred < vocab);
            cout << idToToken(pred, tk, OFFSET_TOKENS) << std::flush;  // Decode tokenid -> token
            tokens.push_back(pred);
            token_pos++;
            // check number of elements in tokens
            // keep a sliding window of maxT tokens at maximum
            if (tokens.size() > static_cast<size_t>(ckpt.maxT)) {
                tokens.erase(tokens.begin());  // remove front element

                if (use_kv_cache) {
                    gpt2.forget_all_tokens();

                    vector<int32_t> prefill_tokens =
                        vector<int32_t>(tokens.begin(), tokens.end() - 1);
                    gpt2.prefill(prefill_tokens);
                    token_pos = tokens.size() - 1;
                }
            }
        } else {
            break;
        }
    }

    auto end_time = std::chrono::steady_clock::now();

    cout << endl
         << "Total Elapsed time: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
         << " milliseconds\n";
}