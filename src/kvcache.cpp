#include "kvcache.hpp"

#include "tensor.hpp"

using tn::Tensor;
using namespace std;

namespace kvcache {

void LayerKVCache::add(Tensor& key, Tensor& value) {
    key_cache_.push_back(std::move(key));
    value_cache_.push_back(std::move(value));
}

void LayerKVCache::get_keys(Tensor& keys) {
    int num_tokens = key_cache_.size();
    keys = Tensor(num_tokens + 1, num_embed_dims_);

    for (int i = 0; i < num_tokens; i++) {
        for (int j = 0; j < num_embed_dims_; j++) {
            keys.set(i, j, key_cache_[i].at(0, j));
        }
    }
}

void LayerKVCache::get_values(Tensor& values) {
    int num_tokens = value_cache_.size();
    values = Tensor(num_tokens + 1, num_embed_dims_);

    for (int i = 0; i < num_tokens; i++) {
        for (int j = 0; j < num_embed_dims_; j++) {
            values.set(i, j, value_cache_[i].at(0, j));
        }
    }
}

int LayerKVCache::get_num_tokens() { return key_cache_.size(); }

void LayerKVCache::remove_oldest_token() {
    if (!key_cache_.empty()) {
        key_cache_.erase(key_cache_.begin());
    }

    if (!value_cache_.empty()) {
        value_cache_.erase(value_cache_.begin());
    }
}

void KVCache::add(int layer_idx, Tensor& key, Tensor& value) {
    kv_cache_[layer_idx].add(key, value);
}

void KVCache::get_keys(int layer_idx, Tensor& keys) { kv_cache_[layer_idx].get_keys(keys); }

void KVCache::get_values(int layer_idx, Tensor& values) { kv_cache_[layer_idx].get_values(values); }

int KVCache::get_num_tokens(int layer_idx) {
    if (num_layers_ > 0) {
        return kv_cache_[layer_idx].get_num_tokens();
    }
    return 0;
}

LayerKVCache& KVCache::get_layer_kv_cache(int layer_idx) { return kv_cache_[layer_idx]; }

void KVCache::remove_oldest_token() {
    for (int i = 0; i < num_layers_; i++) {
        kv_cache_[i].remove_oldest_token();
    }
}

}  // namespace kvcache